"""
Domain & Content Feature Extraction Module
============================================

CRITICAL: The DOMAIN feature extraction logic (features 0-16) MUST be
mirrored EXACTLY in android-app/.../DomainFeatureExtractor.kt for
on-device inference to produce the same results as training.
If you change anything in the domain features, update the Kotlin
counterpart immediately.

Content features (17-22) are extracted during training from page title
and body text provided in final_dataset.csv. On-device, these features
are NOT available at DNS-query time, so the model uses TWO heads:
  - domain-only weights  (17 features, used on-device at DNS time)
  - full weights         (23 features, used only during training
                          evaluation and offline batch testing)

The domain-only model is what ships to the Android app.

Feature vector -- DOMAIN set (17 features, indices 0-16):
  0.  domain_length          - total length of the registrable domain string
  1.  subdomain_count        - number of dot-separated labels minus 2 (0 if apex)
  2.  digit_ratio            - fraction of characters that are digits
  3.  hyphen_count           - number of hyphens
  4.  vowel_ratio            - fraction of alphabetic chars that are vowels
  5.  consonant_run_max      - longest run of consecutive consonants
  6.  shannon_entropy        - character-level entropy (randomness indicator)
  7.  gambling_keyword_hits  - count of known gambling-related substrings
  8.  suspicious_tld         - 1 if TLD is in a commonly abused TLD list, else 0
  9.  digit_letter_transition- number of digit<->letter transitions (e.g. "slot88vip")
  10. length_of_sld          - length of the second-level domain label only
  11. has_random_looking_prefix - 1 if a heuristic flags high-entropy short prefix
  12. has_lucky_number        - 1 if domain contains lucky number sequences common in gambling
  13. has_gambling_compound   - 1 if domain contains compound gambling terms
  14. keyword_density         - ratio of matched keyword characters to SLD length
  15. is_ip_address           - 1 if the domain is a raw IPv4 or IPv6 address
  16. has_compromised_pattern - 1 if domain contains injection/proxy/scan patterns

CONTENT set adds 6 more features (23 total, indices 17-22):
  17. title_keyword_hits      - count of gambling keywords found in page title
  18. title_gambling_density  - ratio of gambling keyword chars to title length
  19. content_keyword_hits    - count of gambling keywords found in page content
  20. content_gambling_density- ratio of gambling keyword chars to content length
  21. has_gambling_title_pattern - 1 if title matches known gambling site naming patterns
  22. content_gambling_vocab_richness - unique gambling keyword types / total vocabulary ratio
"""

import math
import re
from dataclasses import dataclass, asdict

# --- Keyword & TLD lists -----------------------------------------------
# Core explicit gambling terms (safe from generic false positives)
GAMBLING_KEYWORDS = [
    "slot", "gacor", "togel", "toto", "judi", "judol", "bet", "casino", "poker",
    "domino", "qq", "sbobet", "maxwin", "jackpot", "gaple", "rtp",
    "spin", "gambling", "wager", "roulette", "baccarat", "pragmatic",
    "hoki", "menang", "jitu", "prediksi", "taruhan", "bandar", "agen",
    "parlay", "livecasino", "sensasional", "cuan", "rungkat", "rungkad",
    "habanero", "pgsoft", "spadegaming", "zeus", "olympus", "mahjong",
    "bola", "angka", "depo", "wd",
]

# Extended content keywords -- broader set used when scanning page text/title
# Includes Indonesian gambling slang and promotion patterns found in the dataset
CONTENT_GAMBLING_KEYWORDS = GAMBLING_KEYWORDS + [
    "scatter", "freespin", "deposit", "withdraw", "daftar", "login",
    "bonus", "promo", "rebate", "cashback", "turnover", "rollingan",
    "livechat", "whatsapp", "link alternatif", "situs resmi",
    "game online", "judi online", "slot online", "togel online",
    "mix parlay", "bola online", "sportbook", "sportsbook",
    "pulsa", "dana", "gopay", "ovo", "ewallet",
    "gacor hari ini", "bocoran", "pola slot", "jam gacor",
    "winrate", "rtp live", "rtp slot", "provider",
    "pragmatic play", "pg soft", "habanero", "joker123", "joker gaming",
    "live casino", "baccarat", "blackjack", "sicbo", "dragon tiger",
    "cock fight", "sabung ayam", "tembak ikan",
    "4d", "3d", "2d", "colok bebas", "colok jitu", "shio",
    "paito warna", "angka keluaran", "result togel",
    "mpo", "idn", "idnplay", "idnpoker",
    "asia gaming", "sexy gaming", "dream gaming",
]

# Lucky numbers heavily used as branding by gambling sites (e.g. slot88, cuan777, 303)
LUCKY_NUMBERS = [
    "88", "77", "99", "777", "888", "999", "138", "188",
    "303", "5000", "10k", "25k", "8888", "7777",
]

# Compound keywords and indicators common in mirror links and affiliate domains
GAMBLING_COMPOUNDS = [
    "winlink", "linkwin", "linkalt", "alternatif", "linkdaftar", "linklogin",
    "linkgacor", "linkslot", "linkresmi", "megawin", "bigwin", "cuanwin",
    "superwin", "judibola", "depopulsa", "depodana", "agenslot", "bandartogel",
    "slotresmi", "bocoran", "anti-rungkat", "antirungkad", "slotonline",
    "bocorangacor", "rajawin", "kingwin", "hokiterus", "gudangcuan",
]

SUSPICIOUS_TLDS = {
    "xyz", "top", "club", "online", "site", "info", "vip", "biz",
    "live", "fun", "click", "link", "icu", "cc", "me", "pro",
    "asia", "space", "bet", "rest", "buzz", "monster",
}

# Patterns found in compromised/scanned domains (proxy injections, malware C2, SEO spam)
COMPROMISED_PATTERNS = [
    r"[a-z]{2,4}\d{4,}[a-z]{2,4}",          # alphanum run like slot2024win
    r"(slot|togel|judi|casino|bet|gacor|poker|cuan|maxwin)[-_]?(login|daftar|masuk|link|apk|rtp)",
    r"pafi[a-z0-9]+",                       # compromised Indonesian association domains used as gambling landing pages
]

COMPROMISED_RE = re.compile("|".join(COMPROMISED_PATTERNS))

# Title patterns commonly seen in gambling site page titles from the dataset
GAMBLING_TITLE_PATTERNS = [
    re.compile(r"situs\s+(resmi|terpercaya|terbaik)", re.IGNORECASE),
    re.compile(r"(daftar|login|link)\s+(slot|togel|judi|casino|poker)", re.IGNORECASE),
    re.compile(r"(slot|togel|judi|casino)\s+(online|gacor|terpercaya|terbaik)", re.IGNORECASE),
    re.compile(r"agen\s+(slot|togel|judi|bola|casino)", re.IGNORECASE),
    re.compile(r"bandar\s+(slot|togel|judi|bola|casino)", re.IGNORECASE),
    re.compile(r"(rtp|bocoran|pola)\s+(slot|gacor|live|hari ini)", re.IGNORECASE),
    re.compile(r"mix\s*parlay", re.IGNORECASE),
    re.compile(r"(deposit|depo)\s*(pulsa|dana|gopay|ovo|ewallet)", re.IGNORECASE),
    re.compile(r"game\s+(online|gacor|terbaik)\s*(terpercaya)?", re.IGNORECASE),
]

# IPv4 regex
IPV4_RE = re.compile(
    r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$"
)

VOWELS = set("aeiou")


def is_ipv4(s: str) -> bool:
    m = IPV4_RE.match(s)
    if not m:
        return False
    return all(0 <= int(g) <= 255 for g in m.groups())


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in freq.values())


def longest_consonant_run(s: str) -> int:
    s = s.lower()
    max_run = 0
    cur = 0
    for ch in s:
        if ch.isalpha() and ch not in VOWELS:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    return max_run


def digit_letter_transitions(s: str) -> int:
    count = 0
    prev_type = None
    for ch in s:
        if ch.isdigit():
            cur_type = "d"
        elif ch.isalpha():
            cur_type = "l"
        else:
            cur_type = "o"
        if prev_type is not None and cur_type != prev_type and "o" not in (cur_type, prev_type):
            count += 1
        prev_type = cur_type
    return count


def split_domain(domain: str):
    domain = domain.lower().strip(".")
    labels = domain.split(".")
    if len(labels) < 2:
        return labels, labels[0] if labels else "", ""
    tld = labels[-1]
    sld = labels[-2]
    return labels, sld, tld


@dataclass
class DomainFeatures:
    domain_length: int
    subdomain_count: int
    digit_ratio: float
    hyphen_count: int
    vowel_ratio: float
    consonant_run_max: int
    shannon_entropy: float
    gambling_keyword_hits: int
    suspicious_tld: int
    digit_letter_transition: int
    length_of_sld: int
    has_random_looking_prefix: int
    has_lucky_number: int
    has_gambling_compound: int
    keyword_density: float

    def to_vector(self):
        return [
            self.domain_length,
            self.subdomain_count,
            self.digit_ratio,
            self.hyphen_count,
            self.vowel_ratio,
            self.consonant_run_max,
            self.shannon_entropy,
            self.gambling_keyword_hits,
            self.suspicious_tld,
            self.digit_letter_transition,
            self.length_of_sld,
            self.has_random_looking_prefix,
            self.has_lucky_number,
            self.has_gambling_compound,
            self.keyword_density,
        ]


@dataclass
class DomainFeaturesExtended(DomainFeatures):
    """15 base features + 2 extended = 17 total (domain-only)."""
    is_ip_address: int = 0
    has_compromised_pattern: int = 0

    def to_vector(self):
        base = super().to_vector()
        return base + [self.is_ip_address, self.has_compromised_pattern]


@dataclass
class FullFeatures(DomainFeaturesExtended):
    """17 domain features + 6 content features = 23 total.

    Content features are extracted from page title and body text
    available in final_dataset.csv. These are used during training
    for richer learning but NOT available on-device at DNS time.
    """
    title_keyword_hits: int = 0
    title_gambling_density: float = 0.0
    content_keyword_hits: int = 0
    content_gambling_density: float = 0.0
    has_gambling_title_pattern: int = 0
    content_gambling_vocab_richness: float = 0.0

    def to_vector(self):
        base = super().to_vector()
        return base + [
            self.title_keyword_hits,
            self.title_gambling_density,
            self.content_keyword_hits,
            self.content_gambling_density,
            self.has_gambling_title_pattern,
            self.content_gambling_vocab_richness,
        ]


FEATURE_NAMES = [
    "domain_length",
    "subdomain_count",
    "digit_ratio",
    "hyphen_count",
    "vowel_ratio",
    "consonant_run_max",
    "shannon_entropy",
    "gambling_keyword_hits",
    "suspicious_tld",
    "digit_letter_transition",
    "length_of_sld",
    "has_random_looking_prefix",
    "has_lucky_number",
    "has_gambling_compound",
    "keyword_density",
]

FEATURE_NAMES_EXTENDED = FEATURE_NAMES + [
    "is_ip_address",
    "has_compromised_pattern",
]

FEATURE_NAMES_FULL = FEATURE_NAMES_EXTENDED + [
    "title_keyword_hits",
    "title_gambling_density",
    "content_keyword_hits",
    "content_gambling_density",
    "has_gambling_title_pattern",
    "content_gambling_vocab_richness",
]


def _extract_base(domain: str) -> dict:
    """Shared computation for both base and extended feature sets."""
    domain = domain.lower().strip()
    labels, sld, tld = split_domain(domain)

    length = len(domain)
    subdomain_count = max(0, len(labels) - 2)
    digits = sum(c.isdigit() for c in domain)
    digit_ratio = digits / length if length else 0.0
    hyphen_count = domain.count("-")
    alpha_chars = [c for c in domain if c.isalpha()]
    vowel_count = sum(c in VOWELS for c in alpha_chars)
    vowel_ratio = vowel_count / len(alpha_chars) if alpha_chars else 0.0
    consonant_run_max = longest_consonant_run(domain)
    entropy = shannon_entropy(domain)

    # Keyword hits
    matched_keyword_len = 0
    keyword_hits = 0
    for kw in GAMBLING_KEYWORDS:
        if kw in sld:
            keyword_hits += 1
            matched_keyword_len += len(kw)

    # Lucky number detection
    has_lucky_num = 1 if any(num in sld for num in LUCKY_NUMBERS) else 0

    # Compound gambling detection
    has_compound = 0
    for compound in GAMBLING_COMPOUNDS:
        if compound in sld:
            has_compound = 1
            matched_keyword_len += len(compound)
            break

    # Also detect win+link combination (as in gbwinlink.org) or win+lucky_number
    if "win" in sld and ("link" in sld or has_lucky_num == 1):
        has_compound = 1
        matched_keyword_len += 6

    # Keyword density in SLD
    sld_len = len(sld)
    keyword_density = min(1.0, matched_keyword_len / sld_len) if sld_len > 0 else 0.0

    suspicious_tld = 1 if (tld in SUSPICIOUS_TLDS or domain.endswith(".pages.dev") or domain.endswith(".workers.dev")) else 0
    transitions = digit_letter_transitions(sld)

    # Random-looking prefix heuristic (e.g. short high-entropy alphanumeric prefixes)
    random_prefix = 1 if (3 <= sld_len <= 8 and shannon_entropy(sld) >= 2.5 and digits > 0) else 0

    return dict(
        domain=domain, sld=sld, tld=tld, length=length,
        subdomain_count=subdomain_count, digits=digits,
        digit_ratio=digit_ratio, hyphen_count=hyphen_count,
        vowel_ratio=vowel_ratio, consonant_run_max=consonant_run_max,
        entropy=entropy, keyword_hits=keyword_hits,
        matched_keyword_len=matched_keyword_len,
        has_lucky_num=has_lucky_num, has_compound=has_compound,
        sld_len=sld_len, keyword_density=keyword_density,
        suspicious_tld=suspicious_tld, transitions=transitions,
        random_prefix=random_prefix,
    )


def extract_features(domain: str) -> DomainFeatures:
    d = _extract_base(domain)
    return DomainFeatures(
        domain_length=d["length"],
        subdomain_count=d["subdomain_count"],
        digit_ratio=round(d["digit_ratio"], 4),
        hyphen_count=d["hyphen_count"],
        vowel_ratio=round(d["vowel_ratio"], 4),
        consonant_run_max=d["consonant_run_max"],
        shannon_entropy=round(d["entropy"], 4),
        gambling_keyword_hits=d["keyword_hits"],
        suspicious_tld=d["suspicious_tld"],
        digit_letter_transition=d["transitions"],
        length_of_sld=d["sld_len"],
        has_random_looking_prefix=d["random_prefix"],
        has_lucky_number=d["has_lucky_num"],
        has_gambling_compound=d["has_compound"],
        keyword_density=round(d["keyword_density"], 4),
    )


def extract_features_extended(domain: str) -> DomainFeaturesExtended:
    """Extract the full 17-feature vector used for on-device inference."""
    d = _extract_base(domain)

    # Feature 15: is_ip_address (raw IPv4 address)
    ip_flag = 1 if is_ipv4(d["domain"]) else 0

    # Feature 16: has_compromised_pattern
    compromised = 1 if COMPROMISED_RE.search(d["domain"]) else 0

    return DomainFeaturesExtended(
        domain_length=d["length"],
        subdomain_count=d["subdomain_count"],
        digit_ratio=round(d["digit_ratio"], 4),
        hyphen_count=d["hyphen_count"],
        vowel_ratio=round(d["vowel_ratio"], 4),
        consonant_run_max=d["consonant_run_max"],
        shannon_entropy=round(d["entropy"], 4),
        gambling_keyword_hits=d["keyword_hits"],
        suspicious_tld=d["suspicious_tld"],
        digit_letter_transition=d["transitions"],
        length_of_sld=d["sld_len"],
        has_random_looking_prefix=d["random_prefix"],
        has_lucky_number=d["has_lucky_num"],
        has_gambling_compound=d["has_compound"],
        keyword_density=round(d["keyword_density"], 4),
        is_ip_address=ip_flag,
        has_compromised_pattern=compromised,
    )


def _extract_content_features(title: str, content: str) -> dict:
    """Extract features from page title and body content text.

    These features leverage the rich text data in final_dataset.csv
    that was not available in the original domain-only dataset.
    """
    title = str(title).lower() if title else ""
    content = str(content).lower() if content else ""

    # Title keyword analysis
    title_kw_hits = 0
    title_matched_len = 0
    for kw in CONTENT_GAMBLING_KEYWORDS:
        if kw in title:
            title_kw_hits += 1
            title_matched_len += len(kw)

    title_len = max(1, len(title))
    title_gambling_density = min(1.0, title_matched_len / title_len)

    # Content keyword analysis
    content_kw_hits = 0
    content_matched_len = 0
    unique_kw_types = set()
    for kw in CONTENT_GAMBLING_KEYWORDS:
        if kw in content:
            content_kw_hits += 1
            content_matched_len += len(kw)
            unique_kw_types.add(kw)

    content_len = max(1, len(content))
    content_gambling_density = min(1.0, content_matched_len / content_len)

    # Gambling title patterns (regex-based detection of naming conventions)
    has_gambling_title = 0
    for pattern in GAMBLING_TITLE_PATTERNS:
        if pattern.search(title):
            has_gambling_title = 1
            break

    # Vocabulary richness: how diverse are the gambling keywords in the content
    # Normalized by the total keyword vocabulary size
    total_vocab = len(CONTENT_GAMBLING_KEYWORDS)
    vocab_richness = len(unique_kw_types) / total_vocab if total_vocab > 0 else 0.0

    return dict(
        title_kw_hits=title_kw_hits,
        title_gambling_density=round(title_gambling_density, 6),
        content_kw_hits=content_kw_hits,
        content_gambling_density=round(content_gambling_density, 6),
        has_gambling_title=has_gambling_title,
        vocab_richness=round(vocab_richness, 6),
    )


def extract_full_features(domain: str, title: str = "", content: str = "") -> FullFeatures:
    """Extract all 23 features: 17 domain + 6 content.

    Used during training with final_dataset.csv which provides title and content.
    """
    d = _extract_base(domain)

    # Domain extended features
    ip_flag = 1 if is_ipv4(d["domain"]) else 0
    if not ip_flag and re.match(r"^\d{1,3}$", d["sld"]):
        ip_flag = 1
    compromised = 1 if COMPROMISED_RE.search(d["domain"]) else 0

    # Content features
    cf = _extract_content_features(title, content)

    return FullFeatures(
        domain_length=d["length"],
        subdomain_count=d["subdomain_count"],
        digit_ratio=round(d["digit_ratio"], 4),
        hyphen_count=d["hyphen_count"],
        vowel_ratio=round(d["vowel_ratio"], 4),
        consonant_run_max=d["consonant_run_max"],
        shannon_entropy=round(d["entropy"], 4),
        gambling_keyword_hits=d["keyword_hits"],
        suspicious_tld=d["suspicious_tld"],
        digit_letter_transition=d["transitions"],
        length_of_sld=d["sld_len"],
        has_random_looking_prefix=d["random_prefix"],
        has_lucky_number=d["has_lucky_num"],
        has_gambling_compound=d["has_compound"],
        keyword_density=round(d["keyword_density"], 4),
        is_ip_address=ip_flag,
        has_compromised_pattern=compromised,
        title_keyword_hits=cf["title_kw_hits"],
        title_gambling_density=cf["title_gambling_density"],
        content_keyword_hits=cf["content_kw_hits"],
        content_gambling_density=cf["content_gambling_density"],
        has_gambling_title_pattern=cf["has_gambling_title"],
        content_gambling_vocab_richness=cf["vocab_richness"],
    )


if __name__ == "__main__":
    print("=== Domain Feature Extraction Demo ===\n")
    samples = [
        ("gbwinlink.org", True),
        ("slot88vip.xyz", True),
        ("google.com", False),
        ("togelgacor-2024.top", True),
        ("wikipedia.org", False),
        ("mandiri.co.id", False),
        ("linkedin.com", False),
        ("117.18.0.168", True),
        ("137.184.65.66", True),
        ("168pretty.top", True),
    ]
    print(f"{'Domain':<35} {'IP':^5} {'Compr':^5} {'KW':^4} {'Cmpnd':^5} {'Lucky':^5} {'Label'}")
    print("-" * 75)
    for s, expected_gambling in samples:
        f = extract_features_extended(s)
        label = "GAMBLING" if expected_gambling else "legit"
        print(
            f"{s:<35} {f.is_ip_address:^5} {f.has_compromised_pattern:^5} "
            f"{f.gambling_keyword_hits:^4} {f.has_gambling_compound:^5} "
            f"{f.has_lucky_number:^5} {label}"
        )

    print("\n=== Content Feature Extraction Demo ===\n")
    content_samples = [
        (
            "slot88vip.xyz",
            "SLOT88VIP | Situs Slot Gacor Terpercaya Hari Ini",
            "Selamat datang di situs slot gacor terbaik. Daftar sekarang dan dapatkan bonus deposit 100%.",
            True
        ),
        (
            "google.com",
            "Google",
            "Search the world's information, including webpages, images, videos and more.",
            False
        ),
    ]
    print(f"{'Domain':<25} {'TitleKW':^8} {'TitleDens':^10} {'ContentKW':^10} {'ContentDens':^12} {'TitlePat':^9} {'VocabR':^8} {'Label'}")
    print("-" * 100)
    for domain, title, content, expected in content_samples:
        f = extract_full_features(domain, title, content)
        label = "GAMBLING" if expected else "legit"
        print(
            f"{domain:<25} {f.title_keyword_hits:^8} {f.title_gambling_density:^10.4f} "
            f"{f.content_keyword_hits:^10} {f.content_gambling_density:^12.4f} "
            f"{f.has_gambling_title_pattern:^9} {f.content_gambling_vocab_richness:^8.4f} {label}"
        )
