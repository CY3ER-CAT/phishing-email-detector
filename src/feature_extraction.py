"""
feature_extraction.py
=====================

Advanced feature extraction for phishing email detection.

Produces a combined sparse matrix of:

1.  TF-IDF text features (unigrams + bigrams + trigrams) over cleaned text.
2.  Hand-crafted security features:

    * URL-based        : count, suspicious URLs, IP-based, shorteners,
                         HTTP vs HTTPS, subdomains, domain entropy,
                         URL-encoding, @ in host, TLD risk.
    * Text-based       : length, word count, special-char ratio,
                         uppercase ratio, digit ratio, punctuation,
                         exclamation / question / currency counts.
    * Keyword-based    : urgency, authority, scarcity, financial,
                         credential, legal-threat, prize, account-verify,
                         password-reset, delivery, invoice counts.
    * Structural       : has subject-like line, greeting, signature,
                         reply-to mismatch indicator, Flesch readability.

The same extractor is fitted on TRAIN only and reused at prediction time,
preventing data leakage.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import clean_text


# ---------------------------------------------------------------------------
# Keyword dictionaries (lower-case substrings)
# ---------------------------------------------------------------------------

URGENT_WORDS = [
    "urgent", "immediately", "act now", "within 24 hours", "24 hours",
    "expires", "expiration", "deadline", "limited time", "act fast",
    "immediate action", "respond within", "as soon as", "final notice",
    "warning", "suspended", "suspension", "deactivate", "deactivated",
    "closed", "closure", "locked", "lock", "expire today",
]

AUTHORITY_WORDS = [
    "official", "authority", "department", "government", "bank",
    "security team", "fraud department", "compliance", "administrator",
    "system administrator", "support team", "customer service",
]

SCARCITY_WORDS = [
    "limited", "only", "few left", "exclusive", "while supplies last",
    "act before", "expires soon", "last chance", "hurry",
]

FINANCIAL_WORDS = [
    "payment", "bank", "credit card", "debit card", "invoice", "wire transfer",
    "money transfer", "fund", "paypal", "transaction", "refund", "reward",
    "prize", "winner", "winning", "lottery", "inherit", "inheritance",
    "account balance", "direct deposit", "fee", "charge", "bill",
]

CREDENTIAL_WORDS = [
    "password", "username", "login", "sign in", "sign-on", "credential",
    "verify your identity", "confirm identity", "authentication",
    "pin", "otp", "security question", "token",
]

LEGAL_THREAT_WORDS = [
    "legal action", "lawsuit", "court", "police", "arrest", "fine",
    "penalty", "violation", "illegal", "breach", "compromise",
]

ACCOUNT_VERIFICATION_WORDS = [
    "verify your account", "verify account", "confirm your account",
    "update your account", "update account", "security alert",
    "suspicious activity", "unusual activity", "unauthorized access",
    "unauthorised access", "login attempt", "sign-in attempt",
    "verify now", "confirm now", "reactivate",
]

PASSWORD_RESET_WORDS = [
    "reset your password", "password reset", "forgot your password",
    "forgot password", "change your password", "password expired",
    "password change", "reset password", "create a new password",
]

DELIVERY_WORDS = [
    "delivery", "package", "parcel", "shipment", "tracking", "courier",
    "failed delivery", "undelivered", "shipping address",
]

INVOICE_WORDS = [
    "invoice", "receipt", "purchase order", "payment due", "overdue",
    "outstanding balance", "statement", "remittance",
]

SUSPICIOUS_KEYWORDS = (
    URGENT_WORDS + AUTHORITY_WORDS + SCARCITY_WORDS + FINANCIAL_WORDS
    + CREDENTIAL_WORDS + LEGAL_THREAT_WORDS + ACCOUNT_VERIFICATION_WORDS
    + PASSWORD_RESET_WORDS + DELIVERY_WORDS + INVOICE_WORDS
)

SHORTENER_DOMAINS = [
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "buff.ly",
    "adf.ly", "bitly.com", "tiny.cc", "is.gd", "cutt.ly", "rebrand.ly",
    "shorturl.at", "rb.gy", "tinyurl", "bitly",
]

RISKY_TLDS = [
    ".xyz", ".top", ".club", ".loan", ".biz", ".info", ".download",
    ".stream", ".racing", ".review", ".faith", ".date", ".men",
    ".work", ".tech", ".link",
]


# ---------------------------------------------------------------------------
# URL / host helpers
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+", re.IGNORECASE)
_IP_IN_URL_RE = re.compile(
    r"https?://(?:\d{1,3}\.){3}\d{1,3}|www\.(?:\d{1,3}\.){3}\d{1,3}",
    re.IGNORECASE,
)
_PERCENT_ENCODE_RE = re.compile(r"%[0-9a-fA-F]{2}")
_AT_RE = re.compile(r"@")


def extract_urls(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _URL_RE.findall(text)


def _host_of(url: str) -> str:
    m = re.match(r"https?://([^/:?#\s]+)|www\.([^/:?#\s]+)", url, re.IGNORECASE)
    if not m:
        return ""
    return (m.group(1) or m.group(2)).lower()


def domain_entropy(host: str) -> float:
    """Shannon entropy of a hostname (higher = more random-looking)."""
    if not host:
        return 0.0
    host = host.lower()
    probs = [host.count(c) / len(host) for c in set(host)]
    return float(-sum(p * __import__("math").log2(p) for p in probs if p > 0))


def tld_risk(host: str) -> int:
    """1 if the host ends with a risky/free TLD."""
    return int(any(host.endswith(t) for t in RISKY_TLDS))


def count_subdomains(url: str) -> int:
    host = _host_of(url)
    if not host:
        return 0
    return host.count(".")


# ---------------------------------------------------------------------------
# Structural / readability
# ---------------------------------------------------------------------------

_SUBJECT_LIKE_RE = re.compile(r"^(subject|re|fw|fwd):\s*", re.IGNORECASE)
_GREETING_RE = re.compile(r"^(dear|hello|hi|hey|greetings)\b", re.IGNORECASE)
_SIGNATURE_RE = re.compile(r"\b(regards|sincerely|best regards|thank you)\b", re.IGNORECASE)


def _readability(text: str) -> float:
    """Crude Flesch Reading Ease proxy. Returns 0 for empty text."""
    words = len(text.split())
    if words == 0:
        return 0.0
    sentences = max(1, text.count(".") + text.count("!") + text.count("?"))
    syllables = max(1, sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in text.split()))
    return 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)


# ---------------------------------------------------------------------------
# Hand-crafted feature vector
# ---------------------------------------------------------------------------

def extract_handcrafted_features(emails: Sequence[str]) -> np.ndarray:
    """Return dense (n_emails, 30) feature matrix.

    Column order matches ``HANDCRAFTED_FEATURE_NAMES``.
    """
    rows = []
    for email in emails:
        if not isinstance(email, str):
            email = str(email or "")

        urls = extract_urls(email)
        hosts = [_host_of(u) for u in urls]

        num_urls = len(urls)
        num_ip_urls = sum(1 for u in urls if _IP_IN_URL_RE.search(u))
        num_shortened = sum(1 for u in urls if any(d in u.lower() for d in SHORTENER_DOMAINS))
        num_http = sum(1 for u in urls if u.lower().startswith("http://"))
        num_https = sum(1 for u in urls if u.lower().startswith("https://"))
        num_suspicious_urls = sum(
            1 for u in urls
            if _IP_IN_URL_RE.search(u)
            or any(d in u.lower() for d in SHORTENER_DOMAINS)
            or "@" in u
            or "%" in u
            or count_subdomains(u) > 3
            or tld_risk(_host_of(u))
        )
        num_suspicious_url_chars = sum(
            len(_PERCENT_ENCODE_RE.findall(u)) for u in urls
        ) + sum(1 for u in urls if "@" in u)
        excessive_subdomains = sum(1 for u in urls if count_subdomains(u) > 3)
        avg_domain_entropy = float(np.mean([domain_entropy(h) for h in hosts])) if hosts else 0.0
        risky_tld_count = sum(tld_risk(h) for h in hosts)

        low = email.lower()
        email_length = len(email)
        word_count = len(email.split())
        num_special = len(re.findall(r"[^a-zA-Z0-9\s]", email))
        num_uppercase = len(re.findall(r"[A-Z]", email))
        num_digits = len(re.findall(r"\d", email))
        num_exclamation = email.count("!")
        num_question = email.count("?")
        num_currency = len(re.findall(r"[\$£€¥₹]", email))

        def count_cat(words):
            return sum(1 for w in words if w in low)

        features = [
            email_length,
            word_count,
            num_urls,
            num_suspicious_urls,
            num_ip_urls,
            num_shortened,
            num_http,
            num_https,
            excessive_subdomains,
            num_suspicious_url_chars,
            avg_domain_entropy,
            risky_tld_count,
            num_special,
            num_special / max(1, email_length),
            num_uppercase,
            num_uppercase / max(1, email_length),
            num_digits,
            num_exclamation,
            num_question,
            num_currency,
            count_cat(URGENT_WORDS),
            count_cat(AUTHORITY_WORDS),
            count_cat(SCARCITY_WORDS),
            count_cat(FINANCIAL_WORDS),
            count_cat(CREDENTIAL_WORDS),
            count_cat(LEGAL_THREAT_WORDS),
            count_cat(ACCOUNT_VERIFICATION_WORDS),
            count_cat(PASSWORD_RESET_WORDS),
            count_cat(DELIVERY_WORDS),
            count_cat(INVOICE_WORDS),
            int(bool(_SUBJECT_LIKE_RE.search(email.splitlines()[0]) if email.splitlines() else False)),
            int(bool(_GREETING_RE.search(email))),
            int(bool(_SIGNATURE_RE.search(email))),
            max(0.0, _readability(email)),  # clamp readability to non-negative
        ]
        rows.append(features)

    return np.array(rows, dtype=float)


HANDCRAFTED_FEATURE_NAMES = [
    "email_length", "word_count", "num_urls", "num_suspicious_urls",
    "num_ip_urls", "num_shortened_urls", "num_http_urls", "num_https_urls",
    "excessive_subdomains", "num_suspicious_url_chars",
    "avg_domain_entropy", "risky_tld_count", "num_special_chars",
    "special_char_ratio", "num_uppercase", "uppercase_ratio", "num_digits",
    "num_exclamation", "num_question", "num_currency",
    "num_urgent_words", "num_authority_words", "num_scarcity_words",
    "num_financial_words", "num_credential_words", "num_legal_threat_words",
    "num_account_verification_words", "num_password_reset_words",
    "num_delivery_words", "num_invoice_words",
    "has_subject_line", "has_greeting", "has_signature", "readability_score",
]


class FeatureExtractor:
    """TF-IDF + hand-crafted features, fitted on training data only."""

    def __init__(
        self,
        max_features: int = 15000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer: TfidfVectorizer | None = None
        self._fitted = False

    def fit(self, emails: Sequence[str]) -> "FeatureExtractor":
        cleaned = [clean_text(e, keep_urls=False) for e in emails]
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words="english",
        )
        self.vectorizer.fit(cleaned)
        self._fitted = True
        return self

    def transform(self, emails: Sequence[str]) -> sp.csr_matrix:
        if not self._fitted or self.vectorizer is None:
            raise RuntimeError("FeatureExtractor not fitted. Call .fit() first.")
        cleaned = [clean_text(e, keep_urls=False) for e in emails]
        tfidf = self.vectorizer.transform(cleaned)
        hand = sp.csr_matrix(extract_handcrafted_features(emails))
        return sp.hstack([tfidf, hand], format="csr")

    def fit_transform(self, emails: Sequence[str]) -> sp.csr_matrix:
        self.fit(emails)
        return self.transform(emails)

    @property
    def n_features(self) -> int:
        if not self._fitted:
            raise RuntimeError("FeatureExtractor not fitted.")
        return int(self.vectorizer.transform([""]).shape[1]) + len(HANDCRAFTED_FEATURE_NAMES)
