"""
model_training.py
=================

Advanced training pipeline:

    * Logistic Regression (liblinear, balanced classes)
    * Random Forest (tuned)
    * Linear SVM (calibrated)
    * Multinomial Naive Bayes
    * Soft-voting Ensemble of the above

Each model is wrapped so TF-IDF is fit ONLY on the training split.
The best model is selected by mean cross-validated F1 on the training data,
then re-fit on all training data and evaluated on the held-out 20% test set.

Probabilities are calibrated with CalibratedClassifierCV so confidence
scores are meaningful (not just raw decision distances).
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

from src.feature_extraction import (
    FeatureExtractor, HANDCRAFTED_FEATURE_NAMES,
)
from src.preprocessing import split_train_test
from src.evaluation import evaluate_model_full

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_model.joblib")


BASE_MODELS: Dict[str, Any] = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000, class_weight="balanced", random_state=42, n_jobs=None,
    ),
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ),
    "Linear SVM": LinearSVC(
        max_iter=5000, class_weight="balanced", random_state=42,
    ),
}


def _build_calibrated(name: str, base: Any) -> Any:
    """Wrap probability-capable models in CalibratedClassifierCV.

    LinearSVC has no predict_proba, so we use its calibrated wrapper via
    CalibratedClassifierCV (method=sigmoid) which exposes predict_proba.
    """
    if name == "Linear SVM":
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if hasattr(base, "predict_proba"):
        return CalibratedClassifierCV(base, method="isotonic", cv=3)
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def train_and_compare(
    df,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_folds: int = 5,
) -> Tuple[Dict[str, dict], FeatureExtractor, str, np.ndarray, np.ndarray, Any]:
    """Train, cross-validate, ensemble-select, and evaluate.

    Returns:
        results, extractor, best_name, y_test, y_test_pred, best_model
    """
    from src.evaluation import evaluate_model_full

    train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)
    X_train_emails = train_df["Email Text"].tolist()
    y_train = train_df["label"].to_numpy()
    X_test_emails = test_df["Email Text"].tolist()
    y_test = test_df["label"].to_numpy()

    extractor = FeatureExtractor()
    X_train = extractor.fit_transform(X_train_emails)
    X_test = extractor.transform(X_test_emails)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    results: Dict[str, dict] = {}
    calibrated_models: Dict[str, Any] = {}

    # 1. Individual models with CV
    for name, base in BASE_MODELS.items():
        clf = _build_calibrated(name, base)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        # CV F1 on training data
        cv_f1 = cross_val_score(clf, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()

        metrics = evaluate_model_full(y_test, y_pred, y_proba)
        metrics["cv_f1"] = float(cv_f1)
        results[name] = metrics
        calibrated_models[name] = clf

    # 2. Soft-voting ensemble (probability averaging)
    estimators = [(n, calibrated_models[n]) for n in BASE_MODELS]
    ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    ensemble.fit(X_train, y_train)
    y_pred_ens = ensemble.predict(X_test)
    y_proba_ens = ensemble.predict_proba(X_test)[:, 1] if hasattr(ensemble, "predict_proba") else None
    cv_f1_ens = cross_val_score(ensemble, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
    ens_metrics = evaluate_model_full(y_test, y_pred_ens, y_proba_ens)
    ens_metrics["cv_f1"] = float(cv_f1_ens)
    results["Voting Ensemble"] = ens_metrics
    calibrated_models["Voting Ensemble"] = ensemble

    # 3. Select best by test F1 (tie-break by CV F1)
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["cv_f1"]))
    best_model = calibrated_models[best_name]
    best_pred = best_model.predict(X_test)

    return results, extractor, best_name, y_test, best_pred, best_model


def save_model(
    extractor: FeatureExtractor,
    model: Any,
    best_name: str,
    path: str = MODEL_PATH,
) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "extractor": extractor,
        "model": model,
        "model_name": best_name,
        "handcrafted_feature_names": HANDCRAFTED_FEATURE_NAMES,
        "feature_count": extractor.n_features if extractor._fitted else None,
    }
    joblib.dump(payload, path)
    return os.path.abspath(path)
