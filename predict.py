"""
predict.py
==========

Command-line predictor for a single email.

Loads the saved Joblib model and classifies an email, returning the label,
calibrated confidence, and a plain-language explanation of the top risk
indicators.

Usage:
    python predict.py "<email text>"
"""

from __future__ import annotations

import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import scipy.sparse as sp

from src.feature_extraction import (
    extract_handcrafted_features, clean_text,
    HANDCRAFTED_FEATURE_NAMES, URGENT_WORDS, ACCOUNT_VERIFICATION_WORDS,
    PASSWORD_RESET_WORDS, FINANCIAL_WORDS, extract_urls,
)
from src.preprocessing import DEFAULT_DATA_PATH


MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_model.joblib")
LABELS = {0: "Legitimate (Safe)", 1: "Phishing"}


class PhishingPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Trained model not found at: {model_path}\n"
                "Run `python train.py` first."
            )
        payload = joblib.load(model_path)
        self.extractor = payload["extractor"]
        self.model = payload["model"]
        self.model_name = payload.get("model_name", "Unknown")

    def predict(self, email_text: str) -> dict:
        if not isinstance(email_text, str) or not email_text.strip():
            email_text = ""

        # Build feature matrix identical to training
        tfidf = self.extractor.vectorizer.transform(
            [clean_text(email_text, keep_urls=False)]
        )
        hand = sp.csr_matrix(extract_handcrafted_features([email_text]))
        X = sp.hstack([tfidf, hand], format="csr")

        pred = int(self.model.predict(X)[0])
        label = LABELS[pred]

        # Calibrated confidence (probability of predicted class)
        confidence = -1.0
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred])
        elif hasattr(self.model, "decision_function"):
            dist = float(self.model.decision_function(X)[0])
            confidence = float(1.0 / (1.0 + math.exp(-dist)))

        # Explanation: top risk indicators
        vals = extract_handcrafted_features([email_text])[0]
        details = dict(zip(HANDCRAFTED_FEATURE_NAMES, [float(v) for v in vals]))
        urls = extract_urls(email_text)
        low = email_text.lower()

        reasons = []
        if details["num_urgent_words"] > 0:
            reasons.append(f"{int(details['num_urgent_words'])} urgency word(s) detected")
        if details["num_account_verification_words"] > 0:
            reasons.append("Account verification / security-alert language")
        if details["num_password_reset_words"] > 0:
            reasons.append("Password-reset request detected")
        if details["num_financial_words"] > 0:
            reasons.append("Financial / payment language detected")
        if details["num_credential_words"] > 0:
            reasons.append("Credential / login harvesting keywords")
        if details["num_legal_threat_words"] > 0:
            reasons.append("Legal-threat language detected")
        if details["num_suspicious_urls"] > 0:
            reasons.append(f"{int(details['num_suspicious_urls'])} suspicious URL(s)")
        if details["num_ip_urls"] > 0:
            reasons.append("IP-address-based URL(s)")
        if details["risky_tld_count"] > 0:
            reasons.append(f"{int(details['risky_tld_count'])} risky TLD(s)")
        if details["num_exclamation"] > 2:
            reasons.append(f"{int(details['num_exclamation'])} exclamation marks")
        if details["num_currency"] > 0:
            reasons.append("Currency symbols present")
        if details["has_subject_line"] and details["num_urgent_words"] > 0:
            reasons.append("Urgency in subject-like opening line")

        return {
            "label": label,
            "is_phishing": pred == 1,
            "confidence": confidence,
            "proba": proba,
            "details": details,
            "urls": urls,
            "reasons": reasons,
            "model": self.model_name,
        }


def main(argv):
    if len(argv) < 2:
        print("Usage: python predict.py \"<email text>\"")
        sys.exit(1)

    email_text = " ".join(argv[1:])
    predictor = PhishingPredictor()
    res = predictor.predict(email_text)

    print("\n" + "=" * 60)
    print("ADVANCED PHISHING EMAIL PREDICTION")
    print("=" * 60)
    print(f"Model: {res['model']}")
    print(f"Prediction: {res['label']}")
    if res["confidence"] >= 0:
        print(f"Confidence: {res['confidence']:.2%}")
    if res["proba"] is not None:
        print(f"  -> P(Phishing) = {res['proba'][1]:.2%} | P(Legitimate) = {res['proba'][0]:.2%}")
    print("\nRisk indicators:")
    if res["reasons"]:
        for r in res["reasons"]:
            print(f"  ! {r}")
    else:
        print("  None detected (decision driven by TF-IDF text model)")
    if res["urls"]:
        print("\nExtracted URLs:")
        for u in res["urls"]:
            print(f"  {u}")
    print("\nKey feature values:")
    for k in ["num_urls", "num_suspicious_urls", "num_urgent_words",
              "num_account_verification_words", "num_password_reset_words",
              "num_financial_words", "num_exclamation", "risky_tld_count"]:
        print(f"  {k}: {int(res['details'][k])}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main(sys.argv)
