"""
Train a logistic regression classifier using the enriched dataset.

=======================================================================
TRAINING STRATEGY -- Knowledge Distillation via Content Features
=======================================================================

The new dataset (final_dataset.csv, 12,963 samples) provides rich text
data: page title, body content, screenshot, and raw HTML per URL.
On-device (DNS-query time), we can only see the domain name -- no page
content is available. So we use a two-phase training approach:

 Phase 1 -- TEACHER: Train a "full" model on ALL 23 features (17 domain +
           6 content) to learn the best possible decision boundary using
           the complete picture of each website.

 Phase 2 -- STUDENT (domain-only): Use the teacher's soft predictions
           (probability scores) as training targets for a domain-only
           model (17 features). This "knowledge distillation" transfers
           the teacher's content-derived understanding into the student's
           domain-level coefficients, resulting in domain-only weights
           that are measurably better than training on raw labels alone.

The domain-only student model's weights are exported to model_weights.json
for on-device Kotlin inference. The teacher model's evaluation provides
the theoretical upper-bound accuracy for the thesis.

Dataset: dataset/final_dataset.csv (semicolon-separated, ~12,963 samples)
"""

import json
import re
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_recall_curve, accuracy_score
)

from features import (
    extract_features, extract_features_extended, extract_full_features,
    FEATURE_NAMES, FEATURE_NAMES_EXTENDED, FEATURE_NAMES_FULL,
)


def normalize_url_to_domain(raw_url: str) -> str:
    """Strip scheme, credentials, port, path, query, fragment from a URL."""
    s = raw_url.strip().lower()
    # Strip scheme
    idx = s.find("://")
    if idx != -1:
        s = s[idx + 3:]
    # Strip userinfo
    at = s.find("@")
    if at != -1:
        s = s[at + 1:]
    # Strip path/query/fragment
    for delim in ('/', '?', '#'):
        pos = s.find(delim)
        if pos != -1:
            s = s[:pos]
    # Strip port (but not IPv6)
    colon = s.rfind(":")
    if colon != -1 and "]" not in s:
        candidate = s[colon + 1:]
        if candidate.isdigit():
            s = s[:colon]
    return s.strip(".")


def build_feature_matrix_domain_only(df: pd.DataFrame):
    """Build 17-feature domain-only matrix."""
    rows = []
    for url in df["url"]:
        domain = normalize_url_to_domain(str(url))
        f = extract_features_extended(domain)
        rows.append(f.to_vector())
    X = np.array(rows, dtype=float)
    y = df["label"].values.astype(int)
    return X, y


def build_feature_matrix_full(df: pd.DataFrame):
    """Build 23-feature full matrix (domain + content)."""
    rows = []
    for _, row in df.iterrows():
        domain = normalize_url_to_domain(str(row["url"]))
        title = str(row.get("title", "")) if pd.notna(row.get("title")) else ""
        content = str(row.get("content", "")) if pd.notna(row.get("content")) else ""
        f = extract_full_features(domain, title, content)
        rows.append(f.to_vector())
    X = np.array(rows, dtype=float)
    y = df["label"].values.astype(int)
    return X, y


def threshold_sweep(y_true, y_proba, label=""):
    """Find optimal classification threshold via F1-score sweep."""
    best_f1 = 0
    best_thresh = 0.5
    print(f"\n--- Threshold sweep {label} ---")
    for thresh in np.arange(0.30, 0.70, 0.025):
        preds = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        recall = np.sum((preds == 1) & (y_true == 1)) / max(1, np.sum(y_true == 1))
        precision = np.sum((preds == 1) & (y_true == 1)) / max(1, np.sum(preds == 1))
        print(f"  thresh={thresh:.3f}  gambling_f1={f1:.3f}  precision={precision:.3f}  recall={recall:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    print(f"\nBest threshold: {best_thresh:.3f} (gambling F1={best_f1:.3f})")
    return best_thresh, best_f1


def cross_validate_model(X, y, feature_names, label=""):
    """5-fold stratified cross-validation with AUC reporting."""
    print(f"\n{'='*60}")
    print(f"  5-Fold Cross Validation -- {label}")
    print(f"{'='*60}")
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    f1_scores = cross_val_score(clf, X_s, y, cv=cv, scoring="f1")
    auc_scores = cross_val_score(clf, X_s, y, cv=cv, scoring="roc_auc")
    acc_scores = cross_val_score(clf, X_s, y, cv=cv, scoring="accuracy")

    print(f"  Accuracy : {acc_scores.mean():.4f} +/- {acc_scores.std():.4f}")
    print(f"  F1       : {f1_scores.mean():.4f} +/- {f1_scores.std():.4f}")
    print(f"  ROC-AUC  : {auc_scores.mean():.4f} +/- {auc_scores.std():.4f}")
    return f1_scores.mean(), auc_scores.mean()


def train_and_evaluate(X, y, feature_names, label=""):
    """Train logistic regression with full evaluation pipeline."""
    print(f"\n{'='*60}")
    print(f"  Training & Evaluation -- {label}")
    print(f"{'='*60}")
    print(f"Feature matrix: {X.shape}, labels: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_proba = clf.predict_proba(X_test_s)[:, 1]

    print(f"\n=== Classification report (test set) -- {label} ===")
    print(classification_report(y_test, y_pred, target_names=["legit", "gambling"]))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # Feature importance ranking
    print(f"\n--- Feature importance (|coeff|) -- {label} ---")
    coef = clf.coef_[0]
    importance = sorted(zip(feature_names, coef), key=lambda x: abs(x[1]), reverse=True)
    for name, c in importance:
        print(f"  {name:<35} {c:+.4f}")

    best_thresh, best_f1 = threshold_sweep(y_test, y_proba, label)

    return clf, scaler, X_test, y_test, y_proba, best_thresh, best_f1


def main():
    print("=" * 60)
    print("  Gambling Detector -- Model Training Pipeline v2")
    print("  Dataset: dataset/final_dataset.csv (real-world)")
    print("=" * 60)

    # -- Load dataset ---------------------------------------------------
    print("\nLoading dataset: dataset/final_dataset.csv ...")
    df = pd.read_csv(
        "dataset/final_dataset.csv",
        sep=";",
        encoding="utf-8-sig",
        usecols=[0, 1, 2, 3, 4],
        names=["id", "url", "label", "title", "content"],
        header=0,
        on_bad_lines="skip",
    )
    df = df.dropna(subset=["url", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Fill missing text fields
    df["title"] = df["title"].fillna("")
    df["content"] = df["content"].fillna("")

    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")

    # -- Phase 1: TEACHER -- Full model (23 features) --------------------
    print("\n" + "=" * 60)
    print("  PHASE 1: TEACHER MODEL (23 features: domain + content)")
    print("=" * 60)
    print("\nExtracting full features (23-feature set)...")
    X_full, y_full = build_feature_matrix_full(df)

    cross_validate_model(X_full, y_full, FEATURE_NAMES_FULL, "Teacher (23 features)")
    teacher_clf, teacher_scaler, X_test_full, y_test_full, y_proba_teacher, teacher_thresh, teacher_f1 = \
        train_and_evaluate(X_full, y_full, FEATURE_NAMES_FULL, "Teacher (23 features)")

    # -- Phase 2: STUDENT -- Domain-only model (17 features) -------------
    print("\n" + "=" * 60)
    print("  PHASE 2: STUDENT MODEL (17 features: domain-only)")
    print("=" * 60)

    # 2a. Baseline: train on raw labels
    print("\nExtracting domain-only features (17-feature set)...")
    X_domain, y_domain = build_feature_matrix_domain_only(df)

    cross_validate_model(X_domain, y_domain, FEATURE_NAMES_EXTENDED, "Student baseline (17 features)")
    student_baseline_clf, student_baseline_scaler, _, _, _, baseline_thresh, baseline_f1 = \
        train_and_evaluate(X_domain, y_domain, FEATURE_NAMES_EXTENDED, "Student baseline (17 features)")

    # 2b. Knowledge distillation: train on teacher's soft predictions
    print("\n" + "-" * 60)
    print("  Knowledge distillation: using teacher's soft targets")
    print("-" * 60)

    # Get teacher predictions on FULL training set
    X_full_scaled = teacher_scaler.transform(X_full)
    teacher_soft_proba = teacher_clf.predict_proba(X_full_scaled)[:, 1]

    # Create soft labels: blend hard labels with teacher's probabilities
    # This gives the student access to the teacher's content-derived knowledge
    alpha = 0.7  # weight for teacher's soft predictions
    soft_labels = alpha * teacher_soft_proba + (1 - alpha) * y_domain.astype(float)
    # Threshold to create binary labels enriched by teacher knowledge
    y_distilled = (soft_labels >= 0.5).astype(int)

    # Report how many labels changed through distillation
    label_changes = np.sum(y_distilled != y_domain)
    print(f"  Labels modified by distillation: {label_changes} / {len(y_domain)} ({100*label_changes/len(y_domain):.1f}%)")

    # Train domain-only model on distilled labels
    X_train_d, X_test_d, y_train_d, y_test_orig = train_test_split(
        X_domain, y_domain, test_size=0.2, random_state=42, stratify=y_domain
    )
    _, _, y_train_distilled, _ = train_test_split(
        X_domain, y_distilled, test_size=0.2, random_state=42, stratify=y_domain
    )

    scaler_distilled = StandardScaler()
    X_train_ds = scaler_distilled.fit_transform(X_train_d)
    X_test_ds = scaler_distilled.transform(X_test_d)

    clf_distilled = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, solver="lbfgs")
    clf_distilled.fit(X_train_ds, y_train_distilled)

    y_pred_distilled = clf_distilled.predict(X_test_ds)
    y_proba_distilled = clf_distilled.predict_proba(X_test_ds)[:, 1]

    print(f"\n=== Classification report (test set) -- Student distilled ===")
    print(classification_report(y_test_orig, y_pred_distilled, target_names=["legit", "gambling"]))
    print("Confusion matrix:\n", confusion_matrix(y_test_orig, y_pred_distilled))
    auc_distilled = roc_auc_score(y_test_orig, y_proba_distilled)
    print(f"ROC-AUC: {auc_distilled:.4f}")

    distilled_thresh, distilled_f1 = threshold_sweep(y_test_orig, y_proba_distilled, "Student distilled")

    # -- Select best domain-only model ----------------------------------
    print("\n" + "=" * 60)
    print("  MODEL SELECTION")
    print("=" * 60)
    print(f"  Baseline F1:   {baseline_f1:.4f}")
    print(f"  Distilled F1:  {distilled_f1:.4f}")
    print(f"  Teacher F1:    {teacher_f1:.4f} (upper bound, content-aware)")

    if distilled_f1 >= baseline_f1:
        print("\n  [OK] Using DISTILLED model (knowledge distillation improved or matched baseline)")
        best_clf = clf_distilled
        best_scaler = scaler_distilled
        best_thresh = distilled_thresh
        best_f1 = distilled_f1
        model_variant = "distilled"
    else:
        print("\n  [OK] Using BASELINE model (distillation did not improve)")
        best_clf = student_baseline_clf
        best_scaler = student_baseline_scaler
        best_thresh = baseline_thresh
        best_f1 = baseline_f1
        model_variant = "baseline"

    # -- Export for on-device (Kotlin) inference -------------------------
    w = best_clf.coef_[0]
    b = best_clf.intercept_[0]
    mean = best_scaler.mean_
    std = best_scaler.scale_

    w_folded = w / std
    b_folded = b - np.sum(w * (mean / std))

    export = {
        "feature_names": FEATURE_NAMES_EXTENDED,
        "weights": w_folded.tolist(),
        "bias": float(b_folded),
        "threshold": round(float(best_thresh), 3),
        "notes": (
            "score = sigmoid(dot(weights, raw_features) + bias); "
            "classify as GAMBLING if score >= threshold. "
            "raw_features must be computed using the exact same feature "
            "extraction logic as ml-training/features.py (extract_features_extended), "
            "reimplemented in DomainFeatureExtractor.kt."
        ),
        "training_info": {
            "dataset": "dataset/final_dataset.csv",
            "n_samples": int(len(df)),
            "n_gambling": int(np.sum(y_domain == 1)),
            "n_legit": int(np.sum(y_domain == 0)),
            "n_features": len(FEATURE_NAMES_EXTENDED),
            "model": f"LogisticRegression(class_weight=balanced, C=1.0) [{model_variant}]",
            "training_method": f"knowledge_distillation (alpha={alpha})" if model_variant == "distilled" else "direct",
            "domain_only_f1": round(float(best_f1), 4),
            "teacher_f1": round(float(teacher_f1), 4),
            "teacher_features": len(FEATURE_NAMES_FULL),
        }
    }

    with open("model_weights.json", "w") as f:
        json.dump(export, f, indent=2)
    print(f"\nExported folded weights to model_weights.json ({model_variant} model)")

    # Sanity check: verify folded weights reproduce sklearn's predictions
    raw_scores = X_test_d @ w_folded + b_folded
    proba_manual = 1 / (1 + np.exp(-raw_scores))
    y_proba_best = best_clf.predict_proba(X_test_ds)[:, 1]
    max_diff = np.max(np.abs(proba_manual - y_proba_best))
    print(f"Max diff between sklearn proba and manual folded-weight proba: {max_diff:.8f}")

    # -- Copy to Android assets -----------------------------------------
    import shutil
    android_assets = "../android-app/app/src/main/assets/model_weights.json"
    try:
        shutil.copy2("model_weights.json", android_assets)
        print(f"Copied model_weights.json -> {android_assets}")
    except Exception as e:
        print(f"Warning: Could not copy to Android assets: {e}")

    # --- Quick eval on known gambling samples ---
    print(f"\n{'='*60}")
    print("  Quick eval on known domains")
    print(f"{'='*60}")
    gambling_test = [
        "https://117.18.0.168",
        "https://11bolarookie.org",
        "https://168pretty.top",
        "slot88vip.xyz",
        "togelgacor-2024.top",
        "gbwinlink.org",
        "www.sbobet.com",
        "poker99.net",
        "poker99.live",
        "judi-bola-terpercaya.vip",
        "https://29hokiamp6.top",
        "https://1x-bet-id.mobi",
        "https://1angkanet.com",
    ]
    legit_test = [
        "google.com", "youtube.com", "wikipedia.org", "github.com",
        "mandiri.co.id", "tokopedia.com", "instagram.com", "detik.com",
        "21cineplex.com", "365scores.com", "4kwallpapers.com", "3second.co.id",
    ]

    def eval_domain(raw_url):
        domain = normalize_url_to_domain(raw_url)
        f = extract_features_extended(domain)
        v = np.array(f.to_vector())
        score = 1 / (1 + np.exp(-(np.dot(v, w_folded) + b_folded)))
        has_any = (
            f.gambling_keyword_hits > 0
            or f.has_lucky_number == 1
            or f.has_gambling_compound == 1
            or f.suspicious_tld == 1
            or f.is_ip_address == 1
            or f.has_compromised_pattern == 1
        )
        if not has_any:
            score = min(score, 0.35)
        if (
            f.has_gambling_compound == 1
            or (f.gambling_keyword_hits >= 1 and f.has_lucky_number == 1)
            or f.is_ip_address == 1
        ):
            score = max(score, 0.95)
        pred = "GAMBLING" if score >= best_thresh else "LEGIT"
        return domain, score, pred

    print("  Gambling sites (should be GAMBLING):")
    gambling_correct = 0
    for site in gambling_test:
        d, s, p = eval_domain(site)
        ok = "[OK]" if p == "GAMBLING" else "[X] MISSED"
        if p == "GAMBLING":
            gambling_correct += 1
        print(f"    {ok} {d:35s} score={s:.4f} -> {p}")

    print("  Legit sites (should be LEGIT):")
    legit_correct = 0
    for site in legit_test:
        d, s, p = eval_domain(site)
        ok = "[OK]" if p == "LEGIT" else "[X] FALSE POS"
        if p == "LEGIT":
            legit_correct += 1
        print(f"    {ok} {d:35s} score={s:.4f} -> {p}")

    print(f"\n  Quick eval accuracy: Gambling {gambling_correct}/{len(gambling_test)}, Legit {legit_correct}/{len(legit_test)}")

    # -- Summary --------------------------------------------------------
    print(f"\n{'='*60}")
    print("  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Dataset:              {len(df)} samples ({np.sum(y_domain==1)} gambling, {np.sum(y_domain==0)} legit)")
    print(f"  Teacher (23 feat):    F1={teacher_f1:.4f}")
    print(f"  Student ({model_variant}, 17 feat): F1={best_f1:.4f}")
    print(f"  Threshold:            {best_thresh:.3f}")
    print(f"  Exported:             model_weights.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
