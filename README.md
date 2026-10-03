# Gambling Detector — Android On-Device ML DNS Filter

<p align="center">
  <strong>Real-time detection & prevention of online gambling website access on Android endpoints</strong>
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Android-3ddc84?logo=android&logoColor=white">
  <img alt="Min SDK" src="https://img.shields.io/badge/minSdk-26-brightgreen">
  <img alt="Language" src="https://img.shields.io/badge/Kotlin-1.9-7f52ff?logo=kotlin&logoColor=white">
  <img alt="ML" src="https://img.shields.io/badge/ML-Logistic%20Regression-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Getting Started](#getting-started)
  - [ML Training Pipeline](#1-ml-training-pipeline)
  - [Android App](#2-android-app)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Model Performance](#model-performance)
- [Known Limitations](#known-limitations)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

An **intelligent agent-based** Android application that intercepts DNS queries via `VpnService`, classifies queried domains using a **lightweight on-device logistic regression model**, and blocks access to domains predicted to be online gambling sites — **entirely offline, with no data leaving the device**.

This prototype was developed as part of an undergraduate thesis on endpoint-level gambling site detection using machine learning.

### Why DNS-Level Filtering?

| Approach | Pros | Cons |
|----------|------|------|
| Browser extension | Easy to build | Bypassed by using another browser |
| Proxy server | Centralized control | Requires network config, latency |
| **DNS interception (this project)** | **OS-level, app-agnostic, low latency** | Bypassed by DoH/DoT hardcoded DNS |
| Deep packet inspection | Comprehensive | Requires root or network appliance |

---

## Key Features

- 🛡️ **On-device ML classification** — no cloud dependency, no data leaves the phone
- 🌐 **DNS-level interception** — works across all apps (browsers, WebView, etc.)
- 🧠 **17-feature logistic regression** — sub-millisecond inference per query
- 📚 **Knowledge distillation** — trains with 23 features (domain + page content), deploys with 17 (domain-only)
- ✅ **Safe-list guarantee** — zero false positives for whitelisted critical services (Google, banking, government, etc.)
- 🔒 **Fail-secure fallback** — heuristic-based blocking continues even if model weights fail to load
- 📊 **On-device activity log** — SQLite-backed detection history viewable in-app
- 🇮🇩 **Indonesian gambling focus** — keywords, TLDs, and patterns tuned for Indonesian online gambling ecosystem

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                             Android Endpoint                           │
│                                                                         │
│  App DNS query ──▶ system resolver ──▶ 10.111.222.1 (fake DNS address) │
│                                                │                       │
│                          route /32: only this address enters TUN       │
│                                                ▼                       │
│                                    GamblingVpnService TUN fd           │
│                                                │                       │
│                                                ▼                       │
│                                    LocalDnsProxyServer                 │
│                               DomainClassifier.classify()              │
│                                       │              │                 │
│                              gambling │              │ legit           │
│                                       ▼              ▼                 │
│                              NXDOMAIN         Real DNS (8.8.8.8)      │
│                              (blocked)        answer forwarded ✅      │
│                                                                         │
│  ALL other traffic (HTTPS, TCP, …) ──▶ physical NIC ──▶ internet ✅    │
│                    (never enters TUN — no relay needed)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key design decision:** Only the fake DNS IP address (`10.111.222.1/32`) is routed through the TUN interface. All non-DNS traffic bypasses the VPN tunnel entirely, eliminating the need for a TCP relay (like `tun2socks`) and ensuring normal internet speed.

---

## Repository Structure

```
gambling-detector/
│
├── ml-training/                     # Python: dataset, features, training
│   ├── features.py                  # Feature extraction (17 domain + 6 content)
│   ├── train_model.py               # Training pipeline with knowledge distillation
│   ├── test_detector.py             # Test suite mirroring on-device classifier
│   ├── generate_dataset.py          # Dataset validator & analyzer
│   ├── model_weights.json           # Exported model (folded weights + bias)
│   ├── dataset.csv                  # Legacy synthetic dataset
│   └── dataset/                     # Real-world dataset (not in repo — see below)
│       ├── final_dataset.csv        # 12,963 labeled URLs with title & content
│       ├── dataset_html/            # Raw HTML snapshots per URL
│       └── dataset_image/           # Screenshots per URL
│
├── android-app/                     # Android Studio project (Kotlin)
│   ├── app/src/main/
│   │   ├── java/com/thesis/gamblingdetector/
│   │   │   ├── vpn/
│   │   │   │   ├── GamblingVpnService.kt      # TUN setup, VPN lifecycle
│   │   │   │   └── DnsPacketParser.kt         # IPv4/UDP/DNS packet parsing
│   │   │   ├── dns/
│   │   │   │   └── LocalDnsProxyServer.kt     # DNS proxy + classifier bridge
│   │   │   ├── ml/
│   │   │   │   ├── DomainClassifier.kt        # ML inference + hybrid heuristics
│   │   │   │   └── DomainFeatureExtractor.kt  # Feature extraction (mirrors features.py)
│   │   │   ├── data/
│   │   │   │   └── DetectionLogger.kt         # SQLite event logging
│   │   │   └── ui/
│   │   │       └── MainActivity.kt            # Start/stop toggle + activity log
│   │   ├── assets/model_weights.json          # Bundled model weights
│   │   ├── res/layout/activity_main.xml
│   │   └── AndroidManifest.xml
│   ├── build.gradle.kts
│   └── settings.gradle.kts
│
├── .gitignore
├── LICENSE
└── README.md                        # ← You are here
```

---

## System Requirements

### ML Training (Python)

| Component | Requirement |
|-----------|-------------|
| Python | 3.8+ |
| pip packages | `pandas`, `numpy`, `scikit-learn` |
| RAM | ≥ 4 GB (dataset has ~13K rows with text) |
| Dataset | `final_dataset.csv` (see [Dataset Setup](#dataset-setup)) |

### Android App

| Component | Requirement |
|-----------|-------------|
| Android Studio | Hedgehog (2023.1) or newer |
| JDK | 17 |
| Kotlin | 1.9.24 |
| AGP | 8.5.0 |
| Min SDK | 26 (Android 8.0 Oreo) |
| Target SDK | 34 (Android 14) |
| Test device | Physical device recommended (VPN behavior varies on emulators) |

---

## Getting Started

### 1. ML Training Pipeline

#### Install Dependencies

```bash
cd ml-training
pip install pandas numpy scikit-learn
```

#### Dataset Setup

The real-world dataset (`dataset/final_dataset.csv`, ~18 MB CSV + ~4 GB HTML/images) is **not included** in this repository due to size constraints.

**To use the pre-trained model:** The exported `model_weights.json` is already included and ready to use — skip to [Android App](#2-android-app).

**To retrain the model with your own data**, prepare a semicolon-separated CSV file at `ml-training/dataset/final_dataset.csv` with these columns:

```
id;url;label;title;content;image_path;html_path
ID00001;https://example-gambling.xyz;1;Slot Online Gacor;Daftar slot online...;dataset_image/1/ID00001.png;dataset_html/1/ID00001.html
ID00002;https://google.com;0;Google;Search the world...;dataset_image/0/ID00002.png;dataset_html/0/ID00002.html
```

| Column | Description |
|--------|-------------|
| `id` | Unique identifier (e.g. `ID00001`) |
| `url` | Full URL of the website |
| `label` | `1` = gambling, `0` = legitimate |
| `title` | Page title text |
| `content` | Page body text (can be truncated) |
| `image_path` | *(optional)* Path to screenshot |
| `html_path` | *(optional)* Path to raw HTML |

#### Validate Dataset

```bash
python generate_dataset.py
```

This runs integrity checks: label distribution, TLD analysis, content richness, and feature extraction sanity checks.

#### Train the Model

```bash
python train_model.py
```

The training pipeline:
1. **Phase 1 (Teacher):** Trains a 23-feature model (domain + content) as the upper-bound
2. **Phase 2 (Student baseline):** Trains a 17-feature domain-only model on raw labels
3. **Phase 2b (Student distilled):** Trains domain-only on teacher's soft predictions (knowledge distillation)
4. **Selection:** Picks the better student model, exports folded weights to `model_weights.json`
5. **Auto-copy:** Copies weights to `android-app/app/src/main/assets/`

#### Run Tests

```bash
python test_detector.py
```

Validates the exported model against known gambling & legitimate domains, mirroring the exact on-device classification pipeline.

---

### 2. Android App

#### Build & Install

1. Open `android-app/` in **Android Studio**
2. Wait for Gradle sync to complete
3. Connect a **physical Android device** (API 26+) via USB or wireless debugging
4. Click **Run ▶** to build and install

#### First Launch

1. Tap **"Start Protection"**
2. Android shows the standard **VPN consent dialog** — tap OK (required by the OS for any VPN-based app)
3. The notification bar shows a persistent notification indicating active protection
4. DNS queries for gambling domains are now intercepted and blocked

---

## Configuration

### Model Threshold

The classification threshold is stored in `model_weights.json`:

```json
{
  "threshold": 0.45
}
```

- **Lower threshold** (e.g. `0.35`) → more aggressive blocking, higher recall, more false positives
- **Higher threshold** (e.g. `0.60`) → more conservative, fewer false positives, may miss some sites

### Safe-Listed Domains

Critical services that should **never** be blocked are defined in [`DomainClassifier.kt`](android-app/app/src/main/java/com/thesis/gamblingdetector/ml/DomainClassifier.kt):

```kotlin
val SAFE_DOMAINS = setOf(
    "google.com", "youtube.com", "wikipedia.org",
    "mandiri.co.id", "bca.co.id",  // Indonesian banks
    "tokopedia.com", "shopee.co.id", // E-commerce
    // ... and more
)
```

### Upstream DNS Server

By default, legitimate DNS queries are forwarded to **Google DNS (`8.8.8.8:53`)**. This can be changed in `LocalDnsProxyServer.kt`.

---

## Usage

### Testing Detection

After enabling protection, test by visiting gambling-related domains in any browser:

| Domain | Expected Result |
|--------|-----------------|
| `slot88vip.xyz` | 🚫 **Blocked** (gambling keywords + lucky number + suspicious TLD) |
| `gbwinlink.org` | 🚫 **Blocked** (gambling compound detected) |
| `google.com` | ✅ **Allowed** (safe-listed) |
| `wikipedia.org` | ✅ **Allowed** (safe-listed) |
| `mandiri.co.id` | ✅ **Allowed** (safe-listed) |

You can also test from a terminal app on the device:

```bash
# Should fail (blocked — returns NXDOMAIN)
nslookup slot88gacor.xyz

# Should succeed (allowed)
nslookup google.com
```

### Viewing Detection Logs

The **"Recent Activity"** section in the app shows real-time block/allow events with timestamps and classification scores.

---

## How It Works

### Feature Extraction (17 features)

| # | Feature | Description |
|---|---------|-------------|
| 0 | `domain_length` | Total domain string length |
| 1 | `subdomain_count` | Number of subdomain labels |
| 2 | `digit_ratio` | Fraction of digits in domain |
| 3 | `hyphen_count` | Number of hyphens |
| 4 | `vowel_ratio` | Vowel-to-letter ratio |
| 5 | `consonant_run_max` | Longest consonant streak |
| 6 | `shannon_entropy` | Character-level entropy |
| 7 | `gambling_keyword_hits` | Matches against gambling keyword list |
| 8 | `suspicious_tld` | Is TLD in abused-TLD list? |
| 9 | `digit_letter_transition` | Digit↔letter transitions (e.g. `slot88vip`) |
| 10 | `length_of_sld` | Second-level domain length |
| 11 | `has_random_looking_prefix` | High-entropy short prefix flag |
| 12 | `has_lucky_number` | Contains lucky number sequences (`88`, `777`, etc.) |
| 13 | `has_gambling_compound` | Contains compound terms (`winlink`, `megawin`, etc.) |
| 14 | `keyword_density` | Keyword char ratio in SLD |
| 15 | `is_ip_address` | Domain is a raw IPv4 address |
| 16 | `has_compromised_pattern` | Matches compromised/injected domain patterns |

### Classification Pipeline

```
Raw input (URL/domain)
    │
    ▼
Normalize (strip scheme, port, path, www)
    │
    ▼
Safe-list check ──── match? ──▶ ALLOW (score = 0.0)
    │ no match
    ▼
Extract 17 features
    │
    ▼
Logistic Regression: score = σ(w·x + b)
    │
    ▼
Zero-indicator safeguard ──── no signals? ──▶ cap score at 0.35
    │
    ▼
Heuristic booster ──── strong signals? ──▶ boost score to 0.95
    │
    ▼
score ≥ threshold? ──▶ BLOCK (NXDOMAIN) or ALLOW (forward to real DNS)
```

---

## Model Performance

Trained on **12,962 real-world samples** (8,778 legitimate + 4,184 gambling):

| Metric | Domain-Only (on-device) | Full (teacher, 23 features) |
|--------|------------------------|-----------------------------|
| F1-Score | 0.638 | 0.923 |
| Features | 17 | 23 |
| Inference | Sub-millisecond | N/A (training only) |

> **Note:** The domain-only F1 of 0.638 reflects the inherent difficulty of classifying gambling sites from domain names alone (many gambling sites use generic-looking domains). The hybrid heuristic booster and safe-list significantly improve real-world accuracy beyond what the raw F1 suggests.

---

## Known Limitations

1. **DoH/DoT bypass** — Apps using hardcoded DNS-over-HTTPS or DNS-over-TLS bypass the system DNS resolver and are not filtered
2. **IPv6 not handled** — The DNS packet parser only handles IPv4; IPv6-only DNS queries pass through unfiltered
3. **Simplified domain parsing** — No full Public Suffix List; multi-part TLDs like `.co.id` are handled heuristically
4. **Domain-only features** — On-device inference can only use domain name features (no page content at DNS time), limiting accuracy compared to the teacher model
5. **UDP checksum zeroed** — DNS response packets use checksum `0x0000` (valid per RFC 768 for IPv4, but a simplification)

---

## Contributing

Contributions are welcome! Some ideas for improvement:

- [ ] Add IPv6 DNS packet support
- [ ] Integrate a Public Suffix List for robust domain parsing
- [ ] Add SNI-based TLS inspection for HTTPS filtering
- [ ] Pool upstream DNS sockets for lower latency
- [ ] Add periodic model update mechanism
- [ ] Dashboard with detection statistics and charts

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "feat: add my feature"`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <sub>
    Built as part of an undergraduate thesis on intelligent agent-based gambling site detection.
  </sub>
</p>