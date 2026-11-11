"""
test_prediction.py
==================

Automated tests for the Phishing Email Detection System.

Covers:
    * Empty input handling
    * Normal email input
    * URL extraction
    * Prediction output format
    * Model loading
    * Missing dataset handling

Run with:
    python -m pytest tests/
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from src.preprocessing import load_dataset, clean_text
from src.feature_extraction import (
    extract_urls,
    extract_handcrafted_features,
    HANDCRAFTED_FEATURE_NAMES,
    FeatureExtractor,
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "phishing_model.joblib")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")

# ---------------------------------------------------------------------------
# Fixtures / sample emails
# ---------------------------------------------------------------------------

SAFE_EMAIL = (
    "Hello John,\n\n"
    "I hope you are doing well. I am writing to confirm our meeting tomorrow "
    "at 10:00 AM in the conference room. Please let me know if you need to "
    "reschedule.\n\nBest regards,\nSarah"
)

PHISHING_EMAIL = (
    "URGENT: Your account will be suspended!\n\n"
    "Dear customer, we detected suspicious activity on your account. "
    "Please verify your identity immediately by clicking here: "
    "http://bit.ly/3x9fake-secure-login\n\n"
    "Failure to act within 24 hours will result in account closure. "
    "Reset your password now."
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_clean_text_empty_input():
    """Empty or whitespace-only input should not crash and should return empty."""
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    # Non-string input is coerced via str() and must not raise.
    cleaned_none = clean_text(None)
    assert isinstance(cleaned_none, str)


def test_clean_text_removes_html():
    """HTML tags should be stripped from the cleaned text."""
    cleaned = clean_text("<html><body>Hello <b>World</b>!</body></html>")
    assert "<" not in cleaned
    assert ">" not in cleaned
    assert "hello" in cleaned  # clean_text lowercases the text


def test_url_extraction():
    """URLs should be extracted from raw email text."""
    urls = extract_urls(PHISHING_EMAIL)
    assert len(urls) >= 1
    assert any("bit.ly" in u for u in urls)

    # Safe email has no URLs
    assert extract_urls(SAFE_EMAIL) == []


def test_url_extraction_variants():
    """Different URL forms (https, www, IP-based) should be detected."""
    text = (
        "Visit https://secure.example.com/login or www.mybank.net "
        "and http://192.168.1.1/admin"
    )
    urls = extract_urls(text)
    assert len(urls) == 3


def test_handcrafted_features_shape():
    """Feature vector length must match HANDCRAFTED_FEATURE_NAMES."""
    feats = extract_handcrafted_features([SAFE_EMAIL, PHISHING_EMAIL])
    assert feats.shape == (2, len(HANDCRAFTED_FEATURE_NAMES))
    assert (feats >= 0).all()  # all counts are non-negative


def test_handcrafted_features_phishing_higher():
    """Phishing email should flag more risky features than a safe email."""
    safe = extract_handcrafted_features([SAFE_EMAIL])[0]
    phish = extract_handcrafted_features([PHISHING_EMAIL])[0]
    names = HANDCRAFTED_FEATURE_NAMES

    # Phishing should have more URLs and urgent words than safe email.
    assert phish[names.index("num_urls")] > safe[names.index("num_urls")]
    assert phish[names.index("num_urgent_words")] > safe[names.index("num_urgent_words")]


def test_feature_extractor_fit_transform():
    """FeatureExtractor should fit on train and transform test consistently."""
    extractor = FeatureExtractor()
    train = [SAFE_EMAIL, PHISHING_EMAIL, SAFE_EMAIL]
    X_train = extractor.fit_transform(train)
    X_test = extractor.transform([SAFE_EMAIL])
    assert X_train.shape[1] == X_test.shape[1]
    assert X_train.shape[0] == 3
    assert X_test.shape[0] == 1


def test_missing_dataset_handling():
    """Loading a non-existent dataset must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_dataset("/nonexistent/path/to/dataset.csv")


def test_dataset_loads():
    """The shipped dataset should load with expected columns and labels."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("data/dataset.csv not found")
    df = load_dataset(DATA_PATH)
    assert "Email Text" in df.columns
    assert "label" in df.columns
    assert set(df["label"].unique()).issubset({0, 1})
    assert len(df) > 0


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not found")
def test_model_loading():
    """The saved Joblib model should load and contain required keys."""
    import joblib

    payload = joblib.load(MODEL_PATH)
    assert "extractor" in payload
    assert "model" in payload
    assert "model_name" in payload


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not found")
def test_prediction_output():
    """Prediction should return a valid label, confidence, and details dict."""
    sys.path.insert(0, PROJECT_ROOT)
    from predict import PhishingPredictor

    predictor = PhishingPredictor()

    res = predictor.predict(PHISHING_EMAIL)
    assert res["label"] in ("Phishing", "Legitimate (Safe)")
    assert "is_phishing" in res
    if res["confidence"] >= 0:
        assert 0.0 <= res["confidence"] <= 1.0
    assert isinstance(res["details"], dict)
    assert "num_urls" in res["details"]
    assert isinstance(res["reasons"], list)


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="Trained model not found")
def test_empty_input_prediction():
    """Empty input should produce a prediction without crashing."""
    sys.path.insert(0, PROJECT_ROOT)
    from predict import PhishingPredictor

    predictor = PhishingPredictor()
    res = predictor.predict("")
    assert res["label"] in ("Phishing", "Legitimate (Safe)")
    assert "is_phishing" in res
