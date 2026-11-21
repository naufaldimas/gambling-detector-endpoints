"""
Dataset Ingest Helper
=====================

Menambahkan sebuah URL baru ke dataset pelatihan dengan cara:
  1. Mengunduh HTML halaman        -> dataset/dataset_html/<label>/<ID>.html
  2. Mengambil gambar halaman       -> dataset/dataset_image/<label>/<ID>.jpg
       - Prioritas 1: screenshot penuh via Playwright (jika terpasang)
       - Prioritas 2: unduh og:image / twitter:image / favicon dari HTML
       - Jika keduanya gagal: image_path dikosongkan (baris CSV tetap ditulis)
  3. Menambahkan satu baris ke final_dataset.csv dengan format yang sama
     (dipisah ';', ID auto-increment, path relatif gaya Windows).

Modul ini sengaja dibuat tanpa dependency wajib yang berat. HTML diunduh
dengan urllib (bawaan Python). Screenshot hanya aktif jika paket opsional
`playwright` tersedia; selama tidak ada, fitur Add tetap berjalan.

Thread-safety: seluruh operasi penulisan dilindungi satu Lock proses agar
dua request Add yang bersamaan tidak merusak CSV atau menabrak ID.
"""

import os
import re
import csv
import threading
from datetime import datetime
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

# ─── Lokasi dataset (relatif terhadap file ini) ──────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.normpath(os.path.join(_HERE, "..", "ml-training", "dataset"))
CSV_PATH = os.path.join(DATASET_DIR, "final_dataset.csv")
HTML_DIR = os.path.join(DATASET_DIR, "dataset_html")
IMAGE_DIR = os.path.join(DATASET_DIR, "dataset_image")

# Urutan kolom kanonik pada final_dataset.csv
FIELDS = ["id", "url", "label", "title", "content", "image_path", "html_path"]
# Jumlah kolom total pada baris asli (termasuk kolom kosong trailing).
TOTAL_COLUMNS = 27

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
_HTTP_TIMEOUT = 20  # detik

# Satu lock untuk semua penulisan (CSV + penentuan ID) agar aman dari race.
_write_lock = threading.Lock()


# ─── Util ────────────────────────────────────────────────────────────────────

def normalize_url(raw: str) -> str:
    """Pastikan URL punya skema; default-kan ke https."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw):
        raw = "https://" + raw
    return raw


def url_to_domain(raw: str) -> str:
    """Ambil host dari URL (tanpa skema/port/path)."""
    raw = normalize_url(raw)
    try:
        return urlparse(raw).hostname or ""
    except Exception:
        return ""


def _http_get(url: str, timeout: int = _HTTP_TIMEOUT):
    """GET sederhana. Mengembalikan (bytes, content_type, final_url)."""
    req = Request(url, headers={"User-Agent": _USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
        ctype = resp.headers.get("Content-Type", "")
        final_url = resp.geturl()
    return data, ctype, final_url


# ─── Penentuan ID berikutnya ──────────────────────────────────────────────────

_ID_RE = re.compile(r"^ID(\d+)$")


def _next_id() -> str:
    """Tentukan ID berikutnya = max(ID numerik di CSV) + 1, lebar 5 digit.

    Memakai max (bukan jumlah baris) karena ID pada dataset boleh melompat.
    """
    max_n = 0
    if os.path.exists(CSV_PATH):
        with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader, None)
            for row in reader:
                if not row:
                    continue
                m = _ID_RE.match(row[0].strip())
                if m:
                    n = int(m.group(1))
                    if n > max_n:
                        max_n = n
    return f"ID{max_n + 1:05d}"


# ─── Ekstraksi title & content dari HTML ─────────────────────────────────────

def _strip_tags(html: str) -> str:
    # Buang script/style lalu semua tag, rapikan whitespace.
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_title(html: str) -> str:
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return ""


def _extract_meta_image(html: str, base_url: str) -> str:
    """Cari URL gambar dari og:image / twitter:image / <link rel=icon>."""
    patterns = [
        r'(?is)<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'(?is)<link[^>]+rel=["\'][^"\']*icon[^"\']*["\'][^>]+href=["\']([^"\']+)["\']',
    ]
    for pat in patterns:
        m = re.search(pat, html)
        if m:
            return urljoin(base_url, m.group(1).strip())
    return ""


# ─── Penyimpanan gambar ───────────────────────────────────────────────────────

def _try_playwright_screenshot(url: str, out_path: str) -> bool:
    """Coba screenshot via Playwright. True jika berhasil."""
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.goto(url, wait_until="networkidle", timeout=_HTTP_TIMEOUT * 1000)
            page.screenshot(path=out_path, full_page=False)
            browser.close()
        return os.path.exists(out_path) and os.path.getsize(out_path) > 0
    except Exception:
        return False


def _try_download_image(img_url: str, out_path: str) -> bool:
    """Unduh gambar (og:image/favicon). True jika berhasil."""
    if not img_url:
        return False
    try:
        data, ctype, _ = _http_get(img_url)
        if not data or (ctype and "image" not in ctype.lower()):
            return False
        with open(out_path, "wb") as f:
            f.write(data)
        return os.path.getsize(out_path) > 0
    except Exception:
        return False


# ─── API utama ────────────────────────────────────────────────────────────────

def add_url_to_dataset(url: str, label: int) -> dict:
    """Tambahkan satu URL ke dataset. Mengembalikan dict ringkasan hasil.

    Langkah fetch (HTML & gambar) dilakukan di luar lock agar jaringan yang
    lambat tidak memblokir request Add lain. Penentuan ID + penulisan file &
    CSV dilakukan di dalam lock agar konsisten.
    """
    url = normalize_url(url)
    if not url:
        raise ValueError("URL kosong / tidak valid")
    label = int(label)
    if label not in (0, 1):
        raise ValueError("label harus 0 (legit) atau 1 (gambling)")

    # 1) Unduh HTML (sebisanya; kegagalan tidak membatalkan penambahan)
    html_text = ""
    title = ""
    content = ""
    fetch_error = None
    try:
        raw, ctype, final_url = _http_get(url)
        enc = "utf-8"
        m = re.search(r"charset=([\w\-]+)", ctype or "", re.I)
        if m:
            enc = m.group(1)
        html_text = raw.decode(enc, errors="replace")
        title = _extract_title(html_text)
        content = _strip_tags(html_text)[:5000]  # batasi agar CSV tidak membengkak
    except Exception as e:
        fetch_error = f"HTML fetch gagal: {e}"

    # Siapkan folder per-label
    label_dir = str(label)
    html_label_dir = os.path.join(HTML_DIR, label_dir)
    image_label_dir = os.path.join(IMAGE_DIR, label_dir)
    os.makedirs(html_label_dir, exist_ok=True)
    os.makedirs(image_label_dir, exist_ok=True)

    with _write_lock:
        new_id = _next_id()

        # 2) Simpan HTML
        html_rel = ""
        if html_text:
            html_file = os.path.join(html_label_dir, f"{new_id}.html")
            with open(html_file, "w", encoding="utf-8") as f:
                f.write(html_text)
            html_rel = f"dataset_html\\{label_dir}\\{new_id}.html"

        # 3) Simpan gambar (screenshot -> og:image -> kosong)
        image_rel = ""
        image_file = os.path.join(image_label_dir, f"{new_id}.jpg")
        got_image = _try_playwright_screenshot(url, image_file)
        if not got_image and html_text:
            meta_img = _extract_meta_image(html_text, url)
            got_image = _try_download_image(meta_img, image_file)
        if got_image:
            image_rel = f"dataset_image\\{label_dir}\\{new_id}.jpg"

        # 4) Append baris CSV (pertahankan jumlah kolom trailing kosong)
        row = [new_id, url, str(label), title, content, image_rel, html_rel]
        row += [""] * (TOTAL_COLUMNS - len(row))

        need_header = not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0
        with open(CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            if need_header:
                header = list(FIELDS) + [""] * (TOTAL_COLUMNS - len(FIELDS))
                writer.writerow(header)
            writer.writerow(row)

    return {
        "id": new_id,
        "url": url,
        "label": label,
        "title": title,
        "html_path": html_rel,
        "image_path": image_rel,
        "has_html": bool(html_rel),
        "has_image": bool(image_rel),
        "fetch_error": fetch_error,
        "added_at": datetime.now().isoformat(),
    }
