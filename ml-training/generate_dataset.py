"""
Dataset Validator & Analyzer
=============================

Validates and analyzes the real-world dataset (dataset/final_dataset.csv)
ensuring data integrity, consistency, and providing statistical summaries
useful for the thesis methodology chapter.

Previous versions of this script generated SYNTHETIC data. With the
availability of the real labeled dataset (12,963 URLs scraped from
actual gambling and legitimate websites), synthetic generation is no
longer needed. This script now serves as the data preparation and
validation layer.

Dataset structure:
  - id: unique identifier (e.g. ID00001)
  - url: website URL
  - label: 0 (legitimate) or 1 (gambling)
  - title: page title text
  - content: page body text (truncated)
  - image_path: relative path to screenshot
  - html_path: relative path to raw HTML
"""

import csv
import os
import sys
from collections import Counter
from pathlib import Path

# Add parent dir so we can import features
from features import extract_features_extended


DATASET_DIR = Path(__file__).parent / "dataset"
CSV_PATH = DATASET_DIR / "final_dataset.csv"
HTML_DIR = DATASET_DIR / "dataset_html"
IMAGE_DIR = DATASET_DIR / "dataset_image"


def normalize_url_to_domain(raw_url: str) -> str:
    """Strip scheme, credentials, port, path, query, fragment from a URL."""
    s = raw_url.strip().lower()
    idx = s.find("://")
    if idx != -1:
        s = s[idx + 3:]
    at = s.find("@")
    if at != -1:
        s = s[at + 1:]
    for delim in ('/', '?', '#'):
        pos = s.find(delim)
        if pos != -1:
            s = s[:pos]
    colon = s.rfind(":")
    if colon != -1 and "]" not in s:
        candidate = s[colon + 1:]
        if candidate.isdigit():
            s = s[:colon]
    return s.strip(".")


def load_dataset():
    """Load and parse the CSV dataset."""
    if not CSV_PATH.exists():
        print(f"ERROR: Dataset not found at {CSV_PATH}")
        sys.exit(1)

    rows = []
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)
    return rows


def validate_dataset():
    """Run comprehensive validation checks on the dataset."""
    print("=" * 60)
    print("  Dataset Validation & Analysis")
    print(f"  Source: {CSV_PATH}")
    print("=" * 60)

    rows = load_dataset()
    print(f"\n[OK] Loaded {len(rows)} rows")

    # --- Check required columns ---
    required_cols = {"id", "url", "label"}
    optional_cols = {"title", "content", "image_path", "html_path"}
    actual_cols = set(rows[0].keys()) if rows else set()
    missing = required_cols - actual_cols
    if missing:
        print(f"[X] Missing required columns: {missing}")
        return False
    else:
        print(f"[OK] All required columns present: {required_cols}")
    available_optional = optional_cols & actual_cols
    print(f"  Optional columns available: {available_optional}")

    # --- Label distribution ---
    labels = [r.get("label", "").strip() for r in rows]
    label_counts = Counter(labels)
    print(f"\n  Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = 100 * count / len(rows)
        label_name = "Legit" if label == "0" else "Gambling" if label == "1" else f"Unknown({label})"
        print(f"    {label} ({label_name}): {count} ({pct:.1f}%)")

    invalid_labels = {l for l in label_counts if l not in ("0", "1")}
    if invalid_labels:
        print(f"  [X] Invalid labels found: {invalid_labels}")
    else:
        print(f"  [OK] All labels are valid (0 or 1)")

    # --- ID uniqueness ---
    ids = [r["id"] for r in rows]
    if len(ids) == len(set(ids)):
        print(f"  [OK] All {len(ids)} IDs are unique")
    else:
        dupes = [id for id, c in Counter(ids).items() if c > 1]
        print(f"  [X] {len(dupes)} duplicate IDs found: {dupes[:5]}...")

    # --- URL analysis ---
    urls = [r.get("url", "").strip() for r in rows]
    empty_urls = sum(1 for u in urls if not u)
    print(f"\n  URL analysis:")
    print(f"    Total: {len(urls)}, Empty: {empty_urls}")

    domains = [normalize_url_to_domain(u) for u in urls if u]
    unique_domains = set(domains)
    print(f"    Unique domains: {len(unique_domains)}")

    # TLD distribution
    tld_counter = Counter()
    for d in domains:
        parts = d.split(".")
        if len(parts) >= 2:
            tld = parts[-1]
            tld_counter[tld] += 1
    print(f"    Top 10 TLDs:")
    for tld, count in tld_counter.most_common(10):
        print(f"      .{tld}: {count}")

    # IP-based domains
    from features import is_ipv4
    ip_domains = [d for d in domains if is_ipv4(d)]
    ip_labels = []
    for r in rows:
        d = normalize_url_to_domain(r.get("url", ""))
        if is_ipv4(d):
            ip_labels.append(r.get("label", "?"))
    print(f"    IP-based domains: {len(ip_domains)}")
    if ip_labels:
        ip_label_counts = Counter(ip_labels)
        print(f"      Label distribution: {dict(ip_label_counts)}")

    # --- Content richness ---
    title_filled = sum(1 for r in rows if r.get("title", "").strip())
    content_filled = sum(1 for r in rows if r.get("content", "").strip())
    print(f"\n  Content richness:")
    print(f"    Rows with title  : {title_filled}/{len(rows)} ({100*title_filled/len(rows):.1f}%)")
    print(f"    Rows with content: {content_filled}/{len(rows)} ({100*content_filled/len(rows):.1f}%)")

    # --- File existence checks ---
    print(f"\n  File integrity checks:")
    html_exists = 0
    html_missing = 0
    img_exists = 0
    img_missing = 0

    for r in rows[:100]:  # Sample first 100 for speed
        html_path = r.get("html_path", "").strip()
        img_path = r.get("image_path", "").strip()

        if html_path:
            full_html = DATASET_DIR / html_path
            if full_html.exists():
                html_exists += 1
            else:
                html_missing += 1

        if img_path:
            full_img = DATASET_DIR / img_path
            if full_img.exists():
                img_exists += 1
            else:
                img_missing += 1

    print(f"    HTML files (sample 100): {html_exists} found, {html_missing} missing")
    print(f"    Image files (sample 100): {img_exists} found, {img_missing} missing")

    # --- Feature extraction sanity check ---
    print(f"\n  Feature extraction sanity check (sample 5):")
    import random
    random.seed(42)
    sample = random.sample(rows, min(5, len(rows)))
    for r in sample:
        url = r.get("url", "")
        label = r.get("label", "?")
        domain = normalize_url_to_domain(url)
        f = extract_features_extended(domain)
        label_name = "GAMBLING" if label == "1" else "LEGIT"
        print(
            f"    {domain[:30]:<30} label={label_name:<8} "
            f"kw={f.gambling_keyword_hits} compound={f.has_gambling_compound} "
            f"ip={f.is_ip_address} tld_sus={f.suspicious_tld}"
        )

    print(f"\n{'='*60}")
    print("  Validation complete [OK]")
    print(f"{'='*60}")
    return True


if __name__ == "__main__":
    validate_dataset()
