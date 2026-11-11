"""
evaluation.py
=============

Advanced evaluation and visualisation:

    * accuracy, precision, recall, F1, ROC-AUC, PR-AUC
    * confusion-matrix heatmap
    * ROC curve
    * Precision-Recall curve
    * model comparison bar chart
    * permutation feature importance
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    auc,
)
from sklearn.inspection import permutation_importance

from src.feature_extraction import HANDCRAFTED_FEATURE_NAMES, FeatureExtractor


REPORTS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "reports")
)


def evaluate_model_full(y_true, y_pred, y_proba=None) -> dict:
    """Full metric set including ROC-AUC and PR-AUC when probabilities exist."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    return metrics


def classification_report_text(y_true, y_pred) -> str:
    return classification_report(
        y_true, y_pred, target_names=["Legitimate", "Phishing"], zero_division=0
    )


def confusion_matrix_array(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def save_confusion_matrix(y_true, y_pred, filename="confusion_matrix.png") -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    cm = confusion_matrix_array(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=True,
        xticklabels=["Pred. Legitimate", "Pred. Phishing"],
        yticklabels=["Actual Legitimate", "Actual Phishing"],
        linewidths=0.5, linecolor="gray",
        annot_kws={"size": 14, "weight": "bold"},
    )
    plt.title("Confusion Matrix (Test Set)", fontsize=13, weight="bold")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return os.path.abspath(path)


def save_roc_pr_curves(
    y_true, y_proba, filename="roc_pr_curves.png"
) -> str:
    """Combined ROC + Precision-Recall curves."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color="#2ecc71", lw=2,
                 label=f"ROC curve (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], color="#bdc3c7", lw=1, linestyle="--")
    axes[0].set_xlim(0, 1); axes[0].set_ylim(0, 1.02)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve", weight="bold")
    axes[0].legend(loc="lower right")
    axes[0].grid(alpha=0.3)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    axes[1].plot(recall, precision, color="#e74c3c", lw=2,
                 label=f"PR curve (AP = {pr_auc:.3f})")
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1.02)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve", weight="bold")
    axes[1].legend(loc="lower left")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Model Calibration & Discrimination (Test Set)", fontsize=13, weight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return os.path.abspath(path)


def save_model_comparison(results: Dict[str, dict], filename="model_comparison.png") -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = np.arange(len(names))
    width = 0.16

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12", "#9b59b6"]
    for i, metric in enumerate(metrics):
        values = [results[n].get(metric, 0.0) for n in names]
        ax.bar(x + i * width, values, width, label=metric.upper(),
               color=colors[i % len(colors)], edgecolor="black", linewidth=0.4)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=13, weight="bold")
    ax.legend(loc="lower right", ncol=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return os.path.abspath(path)


def _unwrap_estimator(model):
    """Find an inner estimator that exposes ``feature_importances_``.

    Handles CalibratedClassifierCV and VotingClassifier wrappers used by the
    training pipeline. Returns ``None`` when no tree-based estimator exists.
    """
    if hasattr(model, "feature_importances_"):
        return model

    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        for cc in calibrated:
            inner = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
            if inner is not None and hasattr(inner, "feature_importances_"):
                return inner

    estimators = getattr(model, "estimators_", None)
    if estimators:
        for est in estimators:
            inner = _unwrap_estimator(est)
            if inner is not None:
                return inner
    return None


def save_feature_importance(
    model, extractor: FeatureExtractor, X_test, y_test,
    top_n: int = 20, filename="feature_importance.png"
) -> str:
    """Plot the most influential features for the best model.

    Tree ensembles expose ``feature_importances_`` directly, so we use those
    (fast and exact). Other models fall back to permutation importance and —
    because permuting 15k TF-IDF columns is prohibitively slow — the fallback
    is restricted to the 34 handcrafted numeric features.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)

    tfidf_names = list(extractor.vectorizer.get_feature_names_out())
    all_names = tfidf_names + list(HANDCRAFTED_FEATURE_NAMES)

    tree = _unwrap_estimator(model)
    if tree is not None:
        importances = np.asarray(tree.feature_importances_, dtype=float)
        source = "Gini Importance"
    else:
        n_hand = len(HANDCRAFTED_FEATURE_NAMES)
        importances = np.zeros(len(all_names), dtype=float)
        result = permutation_importance(
            model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1,
            scoring="f1",
        )
        importances[-n_hand:] = result.importances_mean
        source = "Permutation Importance (drop in F1)"

    indices = np.argsort(importances)[::-1][:top_n]
    top_names = [all_names[i] for i in indices][::-1]
    top_vals = importances[indices][::-1]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(top_n), top_vals, color="#3498db", edgecolor="black", linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel(source)
    ax.set_title(f"Top {top_n} Most Influential Features", fontsize=13, weight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    path = os.path.join(REPORTS_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(path)


def generate_all_reports(
    y_true, y_pred, y_proba, results: Dict[str, dict],
    model, extractor: FeatureExtractor, X_test,
) -> Tuple[str, str, str]:
    cm_path = save_confusion_matrix(y_true, y_pred)
    curves_path = save_roc_pr_curves(y_true, y_proba)
    comp_path = save_model_comparison(results)
    try:
        fi_path = save_feature_importance(model, extractor, X_test, y_true)
    except Exception:
        fi_path = None
    return cm_path, curves_path, comp_path, fi_path
