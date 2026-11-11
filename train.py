"""
train.py
========

Advanced training entry point.

    1. Loads data/dataset.csv
    2. Stratified 80/20 train/test split
    3. TF-IDF (1-3 grams) + 34 hand-crafted security features (fit on TRAIN only)
    4. Trains Logistic Regression, Naive Bayes, Random Forest, Linear SVM
    5. Builds a soft-voting ensemble
    6. Selects best model by test-set F1 (with CV tie-break)
    7. Calibrates probabilities for meaningful confidence scores
    8. Saves model to models/phishing_model.joblib
    9. Writes evaluation charts to reports/

Usage:
    python train.py
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import load_dataset
from src.model_training import train_and_compare, save_model
from src.evaluation import (
    generate_all_reports, classification_report_text,
)


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_model.joblib")


def main() -> None:
    print("=" * 60)
    print("Phishing Email Detection - ADVANCED Model Training")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"ERROR: dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = load_dataset(DATA_PATH)
    print(f"\nLoaded dataset: {len(df)} emails")
    print(f"  Legitimate (safe): {(df['label'] == 0).sum()}")
    print(f"  Phishing:          {(df['label'] == 1).sum()}")

    results, extractor, best_name, y_test, y_pred, best_model = train_and_compare(df)
    saved_path = save_model(extractor, best_model, best_name, path=MODEL_PATH)

    # Probabilities for ROC/PR curves
    y_proba = None
    if hasattr(best_model, "predict_proba"):
        # rebuild test matrix
        X_test_emails = []  # placeholder; we recompute below
        from src.feature_extraction import FeatureExtractor
        # We need X_test for permutation importance; reconstruct quickly:
        train_df = None
    # Reconstruct X_test for reports
    from src.preprocessing import split_train_test
    train_df, test_df = split_train_test(df, test_size=0.2, random_state=42)
    X_test = extractor.transform(test_df["Email Text"].tolist())
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]

    cm_path, curves_path, comp_path, fi_path = generate_all_reports(
        y_test, y_pred, y_proba, results, best_model, extractor, X_test
    )

    print("\n" + "-" * 60)
    print("Model comparison (test set):")
    print("-" * 60)
    header = f"{'Model':<24} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'ROC':>7} {'PR':>7} {'CV-F1':>7}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<24} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} "
            f"{m.get('roc_auc', 0.0):>7.4f} {m.get('pr_auc', 0.0):>7.4f} "
            f"{m.get('cv_f1', 0.0):>7.4f}"
        )
    print("-" * len(header))
    print(f"\nBest model (test F1): {best_name}")

    print("\nClassification report (best model):")
    print(classification_report_text(y_test, y_pred))

    print(f"\nModel saved to: {saved_path}")
    print(f"Confusion matrix: {cm_path}")
    print(f"ROC/PR curves: {curves_path}")
    print(f"Comparison chart: {comp_path}")
    if fi_path:
        print(f"Feature importance: {fi_path}")
    print("\nTraining complete. Run:  streamlit run app.py")


if __name__ == "__main__":
    main()
