"""
preprocessing.py
================

Advanced data loading and text-cleaning pipeline for the Phishing Email
Detection System.

Includes:
    * NLTK-based tokenisation, stop-word removal and lemmatisation
    * HTML stripping and URL masking (configurable)
    * Safe stratified train/test split with leakage prevention
    * Dataset validation helpers
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# NLTK bootstrap with graceful fallback (no network required)
# ---------------------------------------------------------------------------

_NLTK_OK = False
try:
    nltk.data.find("tokenizers/punkt")
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
    _NLTK_OK = True
except LookupError:
    _NLTK_OK = False


# Built-in fallback stop-word list (common English function words).
_FALLBACK_STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any",
    "are","aren't","as","at","be","because","been","before","being","below",
    "between","both","but","by","can't","cannot","could","couldn't","did",
    "didn't","do","does","doesn't","doing","don't","down","during","each",
    "few","for","from","further","had","hadn't","has","hasn't","have","haven't",
    "having","he","he'd","he'll","he's","her","here","here's","hers","herself",
    "him","himself","his","how","how's","i","i'd","i'll","i'm","i've","if",
    "in","into","is","isn't","it","it's","its","itself","let's","me","more",
    "most","mustn't","my","myself","no","nor","not","of","off","on","once",
    "only","or","other","ought","our","ours","ourselves","out","over","own",
    "same","shan't","she","she'd","she'll","she's","should","shouldn't","so",
    "some","such","than","that","that's","the","their","theirs","them",
    "themselves","then","there","there's","these","they","they'd","they'll",
    "they're","they've","this","those","through","to","too","under","until",
    "up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
    "weren't","what","what's","when","when's","where","where's","which",
    "while","who","who's","whom","why","why's","with","won't","would",
    "wouldn't","you","you'd","you'll","you're","you've","your","yours",
    "yourself","yourselves",
}


def _get_stopwords() -> set:
    if _NLTK_OK:
        return set(stopwords.words("english"))
    return set(_FALLBACK_STOPWORDS)


def _get_lemmatizer():
    if _NLTK_OK:
        return WordNetLemmatizer()
    return None


_STOPWORDS = _get_stopwords()
_LEMMATIZER = _get_lemmatizer()


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DEFAULT_DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the labelled email dataset from CSV.

    Expected columns:
        - ``Email Text`` : raw email body
        - ``label``      : 1 = phishing, 0 = legitimate
        - ``Email Type`` : human-readable label

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if required columns are missing or labels are invalid.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Place a dataset.csv in the data/ folder or run the data prep step."
        )

    df = pd.read_csv(path)

    required = {"Email Text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.copy()
    df["Email Text"] = df["Email Text"].astype(str).fillna("")
    df["label"] = df["label"].astype(int)

    invalid = set(df["label"].unique()) - {0, 1}
    if invalid:
        raise ValueError(f"label column contains invalid values: {invalid}")

    return df


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

URL_RE = re.compile(r"https?://[^\s<>\"]+|www\.[^\s<>\"]+", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_ALPHANUM_RE = re.compile(r"[^a-zA-Z\s]")


def remove_html_tags(text: str) -> str:
    """Strip HTML/XML tags, keeping visible text."""
    if not text:
        return ""
    text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    text = HTML_TAG_RE.sub(" ", text)
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&nbsp;": " ", "&apos;": "'",
    }
    for ent, char in entities.items():
        text = text.replace(ent, char)
    return text


def clean_text(
    text: str,
    *,
    keep_urls: bool = False,
    lemmatize: bool = True,
    remove_stopwords: bool = True,
) -> str:
    """Advanced cleaning for a single email body.

    Steps:
        1. Strip HTML tags
        2. Optionally mask URLs with the token `` URL `` (kept if keep_urls)
        3. Lower-case
        4. Remove non-alphanumeric characters
        5. Tokenise, lemmatise, remove stop-words
        6. Re-join into a single string
    """
    if not isinstance(text, str):
        text = str(text or "")

    text = remove_html_tags(text)

    if not keep_urls:
        text = URL_RE.sub(" URL ", text)

    text = text.lower()
    text = NON_ALPHANUM_RE.sub(" ", text)

    if _NLTK_OK:
        tokens = word_tokenize(text)
    else:
        tokens = [t for t in text.split() if t]

    if remove_stopwords:
        tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if lemmatize and _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def clean_emails(emails: List[str], **kwargs) -> List[str]:
    """Clean a list of email bodies."""
    return [clean_text(e, **kwargs) for e in emails]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split preserving class distribution.

    Stratification guarantees the phishing : legitimate ratio is identical in
    both splits and prevents evaluation bias / leakage.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
