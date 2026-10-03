"""
Test Suite for Gambling Domain Detection (v2)
=============================================
Tests the model against real-world URLs from final_dataset.csv
and known edge cases, using the exported model_weights.json.
Validates the complete hybrid classifier pipeline (safe-list,
ML scoring, zero-indicator safeguard, and heuristic booster)
mirroring DomainClassifier.kt.
"""
import json
import numpy as np
from features import extract_features_extended


SAFE_DOMAINS = {
    "google.com", "gstatic.com", "googleapis.com", "googleusercontent.com",
    "youtube.com", "ytimg.com", "googlevideo.com",
    "wikipedia.org", "wikimedia.org", "github.com", "githubusercontent.com",
    "microsoft.com", "apple.com", "amazon.com", "cloudflare.com",
    "linkedin.com", "windows.com", "live.com", "office.com",
    "mandiri.co.id", "bankmandiri.co.id", "bca.co.id", "klikbca.com",
    "bri.co.id", "bni.co.id", "cimbniaga.co.id", "dana.id", "ovo.id",
    "kemdikbud.go.id", "kominfo.go.id", "go.id", "ac.id",
    "detik.com", "kompas.com", "tempo.co", "tribunnews.com",
    "tokopedia.com", "shopee.co.id", "bukalapak.com", "gojek.com", "grab.com",
    "whatsapp.com", "whatsapp.net", "instagram.com", "facebook.com", "fbcdn.net",
    "twitter.com", "x.com", "tiktok.com", "byteoversea.com",
}


def classify_domain(clean_domain: str, weights, bias, threshold):
    """Mirror DomainClassifier.kt on-device inference logic."""
    # 1. Safe-list check
    for safe in SAFE_DOMAINS:
        if clean_domain == safe or clean_domain.endswith("." + safe):
            return 0.0, False, "Safe-listed"

    # 2. Extract features
    feat_obj = extract_features_extended(clean_domain)
    vec = np.array(feat_obj.to_vector())

    # Pad or trim to match model weights
    if len(vec) < len(weights):
        vec = np.pad(vec, (0, len(weights) - len(vec)))
    elif len(vec) > len(weights):
        vec = vec[:len(weights)]

    # 3. Logistic Regression scoring
    z = np.dot(weights, vec) + bias
    score = 1.0 / (1.0 + np.exp(-z))

    # 4. Zero-indicator safeguard:
    # A domain without ANY gambling keywords, lucky numbers, compounds,
    # suspicious TLD, direct IP, or compromised pattern should not be
    # classified as gambling due to incidental digits or length.
    has_any = (
        feat_obj.gambling_keyword_hits > 0
        or feat_obj.has_lucky_number == 1
        or feat_obj.has_gambling_compound == 1
        or feat_obj.suspicious_tld == 1
        or feat_obj.is_ip_address == 1
        or feat_obj.has_compromised_pattern == 1
    )
    if not has_any:
        score = min(score, 0.35)

    # 5. Hybrid Heuristic Booster:
    # Strong signals are boosted to ensure decisive blocking
    if (
        feat_obj.has_gambling_compound == 1
        or (feat_obj.gambling_keyword_hits >= 1 and feat_obj.has_lucky_number == 1)
        or feat_obj.is_ip_address == 1
    ):
        score = max(score, 0.95)

    is_gambling = score >= threshold
    return score, is_gambling, "ML+Hybrid"


def run_tests():
    with open('model_weights.json') as f:
        model = json.load(f)

    weights = np.array(model['weights'])
    bias = model['bias']
    thresh = model['threshold']
    n_features = len(weights)

    print(f"Model: {n_features} features, threshold={thresh}")
    print(f"Training info: {json.dumps(model.get('training_info', {}), indent=2)}\n")

    # -----------------------------------------------------------------
    # Test cases from the REAL dataset (final_dataset.csv)
    # -----------------------------------------------------------------
    test_cases = [
        # === GAMBLING domains (from dataset label=1) ===
        ('117.18.0.168', 1),                    # IP-based gambling site (PARLAY855)
        ('11bolarookie.org', 1),                 # 11BOLA slot site
        ('168pretty.top', 1),                    # ATM4D gambling login
        ('137.184.65.66', 1),                    # MAMI188 IP-based site
        ('165.22.53.103', 1),                    # DORAHOKI IP-based site
        ('1angkanet.com', 1),                    # Togel/paito warna site
        ('1x-bet-id.mobi', 1),                   # 1xBet betting site
        ('1x-bet.mobi', 1),                      # 1xBet variant
        ('29hokiamp6.top', 1),                    # 29Hoki slot/casino
        ('69aladashvili-en.pages.dev', 1),       # MAHJONG900 togel
        ('700tea.xyz', 1),                        # MPO700 slot site
        ('33pgxyz33007890055.shop', 0),          # Google phishing page (labeled 0)

        # === GAMBLING domains (synthetically constructed test cases) ===
        ('gbwinlink.org', 1),
        ('slot88gacor.xyz', 1),
        ('judibola-resmi.vip', 1),
        ('sbobet138.net', 1),
        ('linkalternatif-slot.org', 1),
        ('cuan777.top', 1),
        ('zeusolympus-slot.club', 1),
        ('depopulsa10k.site', 1),
        ('bandartogel88.asia', 1),
        ('maxwin-sensasional.bet', 1),

        # === LEGIT domains (from dataset label=0) ===
        ('1001fonts.com', 0),                     # Font download site
        ('10fastfingers.com', 0),                 # Typing test site
        ('11zon.com', 0),                         # PDF tool site
        ('168railway.com', 0),                    # Train schedule site
        ('1830inn.com', 0),                       # Mackinac Island B&B
        ('192-168-1-1ip.mobi', 0),               # Router config guide
        ('19216811.uno', 0),                      # Router IP guide
        ('19216811ip.host', 0),                   # Router IP guide
        ('21cineplex.com', 0),                    # Cinema 21 Indonesia
        ('23paskal.id', 0),                       # Shopping center
        ('24live.com', 0),                        # Live scores
        ('365scores.com', 0),                     # Sports scores
        ('3second.co.id', 0),                     # Fashion brand
        ('4kwallpapers.com', 0),                  # Wallpaper site
        ('4shared.com', 0),                       # File sharing
        ('50topburger.com', 0),                   # Food portal
        ('google.com', 0),
        ('youtube.com', 0),
        ('mandiri.co.id', 0),
        ('bca.co.id', 0),
        ('detik.com', 0),
        ('kompas.com', 0),
        ('linkedin.com', 0),
        ('wikipedia.org', 0),
        ('github.com', 0),
        ('kemdikbud.go.id', 0),
        ('tokopedia.com', 0),
        ('shopee.co.id', 0),
    ]

    print("=" * 75)
    print(f"{'Domain':<35} | {'Expected':<8} | {'Score':<7} | {'Result'}")
    print("=" * 75)

    all_passed = True
    false_positives = []
    false_negatives = []
    total_gambling = 0
    total_legit = 0
    correct_gambling = 0
    correct_legit = 0

    for dom, exp in test_cases:
        score, is_gambling, mode = classify_domain(dom, weights, bias, thresh)
        status = 'PASS' if (is_gambling == bool(exp)) else 'FAIL'

        if status == 'FAIL':
            all_passed = False
            if exp == 0 and is_gambling:
                false_positives.append(dom)
            elif exp == 1 and not is_gambling:
                false_negatives.append(dom)

        if exp == 1:
            total_gambling += 1
            if is_gambling:
                correct_gambling += 1
        else:
            total_legit += 1
            if not is_gambling:
                correct_legit += 1

        res_str = 'BLOCKED' if is_gambling else 'ALLOWED'
        exp_str = 'GAMBLING' if exp else 'LEGIT'
        marker = '[OK]' if status == 'PASS' else '[X]'
        print(f"{marker} {dom:<33} | {exp_str:<8} | {score:.4f}  | {res_str} ({status})")

    print("=" * 75)

    # Summary
    print(f"\n{'-'*40}")
    print(f"  RESULTS SUMMARY")
    print(f"{'-'*40}")
    print(f"  Gambling detection : {correct_gambling}/{total_gambling} ({100*correct_gambling/max(1,total_gambling):.0f}%)")
    print(f"  Legit detection    : {correct_legit}/{total_legit} ({100*correct_legit/max(1,total_legit):.0f}%)")
    print(f"  False Positives    : {len(false_positives)}")
    print(f"  False Negatives    : {len(false_negatives)}")

    if false_positives:
        print(f"\n  FP domains: {', '.join(false_positives)}")
    if false_negatives:
        print(f"  FN domains: {', '.join(false_negatives)}")

    if all_passed:
        print("\n[OK] ALL TESTS PASSED!")
    else:
        print(f"\n[X] {len(false_positives) + len(false_negatives)} TESTS FAILED -- needs investigation")


if __name__ == '__main__':
    run_tests()
