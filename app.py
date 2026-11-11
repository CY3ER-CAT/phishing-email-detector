"""
app.py
======

Advanced Streamlit dashboard for the Phishing Email Detection System.

Tabs:
    A. Home Dashboard      - key metrics & dataset overview
    B. Email Analyzer      - single-email analysis with header parsing,
                             URL extraction, risk explanation
    C. Batch Analyzer      - upload a CSV of emails for bulk scoring
    D. Model Performance   - metrics, ROC/PR curves, confusion matrix,
                             feature importance, model comparison
    E. Dataset Information - dataset source, counts, train/test split

Run:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import io

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image

import joblib
import scipy.sparse as sp

from src.preprocessing import load_dataset, DEFAULT_DATA_PATH, clean_text
from src.feature_extraction import (
    extract_handcrafted_features, extract_urls,
    HANDCRAFTED_FEATURE_NAMES,
)
from src.model_training import MODEL_PATH

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Phishing Email Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark, professional custom CSS
st.markdown(
    """
    <style>
    .main {background: #0f172a; color: #e2e8f0;}
    .stApp {background: #0f172a;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    h1, h2, h3 {color: #f8fafc;}
    .metric-card {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 1rem; margin-bottom: 1rem;
    }
    .metric-value {font-size: 1.75rem; font-weight: 700; color: #38bdf8;}
    .metric-label {font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;}
    .badge-phish {
        display:inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        background: #7f1d1d; color: #fecaca; font-weight: 700; font-size: 0.9rem;
    }
    .badge-safe {
        display:inline-block; padding: 0.25rem 0.75rem; border-radius: 999px;
        background: #14532d; color: #bbf7d0; font-weight: 700; font-size: 0.9rem;
    }
    .stButton>button {
        background: #2563eb; color: white; border-radius: 6px; font-weight: 600;
    }
    .stButton>button:hover {background: #1d4ed8;}
    .sidebar .sidebar-content {background: #1e293b;}
    .divider {border-top: 1px solid #334155; margin: 1rem 0;}
    .small {font-size: 0.8rem; color: #94a3b8;}
    </style>
    """,
    unsafe_allow_html=True,
)

LABELS = {0: "Legitimate (Safe)", 1: "Phishing"}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "dataset.csv")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return load_dataset(path)


@st.cache_resource(show_spinner=False)
def load_model_payload(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


# ---------------------------------------------------------------------------
# Core prediction logic (shared by single + batch)
# ---------------------------------------------------------------------------

def predict_email(email_text: str, payload: dict) -> dict:
    """Run prediction + explanation for one email string."""
    extractor = payload["extractor"]
    model = payload["model"]
    model_name = payload.get("model_name", "Unknown")

    tfidf = extractor.vectorizer.transform([clean_text(email_text, keep_urls=False)])
    hand = sp.csr_matrix(extract_handcrafted_features([email_text]))
    X = sp.hstack([tfidf, hand], format="csr")

    pred = int(model.predict(X)[0])
    label = LABELS[pred]

    confidence = -1.0
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(proba[pred])

    vals = extract_handcrafted_features([email_text])[0]
    details = dict(zip(HANDCRAFTED_FEATURE_NAMES, [float(v) for v in vals]))
    urls = extract_urls(email_text)
    low = email_text.lower()

    reasons = []
    if details["num_urgent_words"] > 0:
        reasons.append(f"{int(details['num_urgent_words'])} urgency word(s)")
    if details["num_account_verification_words"] > 0:
        reasons.append("Account verification / security-alert language")
    if details["num_password_reset_words"] > 0:
        reasons.append("Password-reset request")
    if details["num_financial_words"] > 0:
        reasons.append("Financial / payment language")
    if details["num_credential_words"] > 0:
        reasons.append("Credential / login harvesting keywords")
    if details["num_legal_threat_words"] > 0:
        reasons.append("Legal-threat language")
    if details["num_suspicious_urls"] > 0:
        reasons.append(f"{int(details['num_suspicious_urls'])} suspicious URL(s)")
    if details["num_ip_urls"] > 0:
        reasons.append("IP-address-based URL(s)")
    if details["risky_tld_count"] > 0:
        reasons.append(f"{int(details['risky_tld_count'])} risky TLD(s)")
    if details["num_exclamation"] > 2:
        reasons.append(f"{int(details['num_exclamation'])} exclamation marks")
    if details["num_currency"] > 0:
        reasons.append("Currency symbol(s) present")

    return {
        "label": label,
        "is_phishing": pred == 1,
        "confidence": confidence,
        "proba": proba,
        "details": details,
        "urls": urls,
        "reasons": reasons,
        "model": model_name,
    }


def parse_headers(raw_email: str) -> dict:
    """Very light header extraction: looks for common header lines at the top."""
    headers = {}
    for line in raw_email.splitlines()[:40]:
        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip().lower()
            if key in ("from", "to", "subject", "date", "reply-to", "return-path", "received", "x-mailer", "message-id"):
                headers[key] = val.strip()
    return headers


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🛡️ Phishing Detector")
    st.caption("Advanced ML Email Security Platform")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    st.subheader("Navigation")
    page = st.radio(
        "Go to",
        [
            "A. Home Dashboard",
            "B. Email Analyzer",
            "C. Batch Analyzer",
            "D. Model Performance",
            "E. Dataset Information",
        ],
        label_visibility="collapsed",
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.subheader("Model status")
    if os.path.exists(MODEL_PATH):
        payload = load_model_payload(MODEL_PATH)
        st.success("✅ Trained model loaded")
        st.write(f"**Algorithm:** {payload.get('model_name', 'Unknown') if payload else 'Unknown'}")
    else:
        st.error("❌ No trained model found")
        st.write("Run `python train.py` first.")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.caption(
        "⚠️ This is a machine-learning detection aid, **not a guarantee** "
        "that an email is safe. Emails are processed locally — nothing is "
        "sent to external services and no links are opened."
    )

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = None
if os.path.exists(DATA_PATH):
    try:
        df = load_data(DATA_PATH)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
else:
    st.warning(f"Dataset not found at {DATA_PATH}")

# ---------------------------------------------------------------------------
# Tab A: Home Dashboard
# ---------------------------------------------------------------------------

if page == "A. Home Dashboard":
    st.title("📊 Phishing Email Detection System")
    st.subheader("Home Dashboard")

    st.info(
        "🔴 **Phishing** emails attempt to steal credentials or money. "
        "🟢 **Safe (Legitimate)** emails are genuine. This dashboard summarises "
        "the dataset and the trained detection model."
    )

    if df is not None:
        total = len(df)
        phishing = int((df["label"] == 1).sum())
        safe = int((df["label"] == 0).sum())

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-label">Total emails in dataset</div>'
                        f'<div class="metric-value">{total:,}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-label">Safe emails</div>'
                        f'<div class="metric-value">{safe:,}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-label">Phishing emails</div>'
                        f'<div class="metric-value">{phishing:,}</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown('<div class="metric-card">'
                        f'<div class="metric-label">Emails analysed (session)</div>'
                        f'<div class="metric-value">{st.session_state.analysis_count}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Class distribution")
        col_chart, col_note = st.columns([2, 1])
        with col_chart:
            st.bar_chart(
                pd.DataFrame(
                    {"Count": [safe, phishing]},
                    index=["Legitimate (Safe)", "Phishing"],
                )
            )
        with col_note:
            st.write(
                f"- **Legitimate (Safe):** {safe:,}  ({safe / total:.1%})\n"
                f"- **Phishing:** {phishing:,}  ({phishing / total:.1%})"
            )
            st.caption("Classes preserved via stratified 80/20 train/test split.")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Trained model")
        if os.path.exists(MODEL_PATH):
            payload = load_model_payload(MODEL_PATH)
            st.write(f"**Algorithm:** {payload.get('model_name', 'Unknown')}")
            st.write(f"**Model file:** `models/phishing_model.joblib`")
        else:
            st.warning("No trained model found. Train via `python train.py`.")
    else:
        st.error("Dataset could not be loaded.")

# ---------------------------------------------------------------------------
# Tab B: Email Analyzer
# ---------------------------------------------------------------------------

elif page == "B. Email Analyzer":
    st.title("✉️ Email Analyzer")
    st.subheader("Paste an email below to check whether it is phishing or safe.")

    st.warning(
        "🔒 **Security:** processed locally. No links opened, no external "
        "calls, nothing stored. Treat all input as untrusted."
    )

    include_headers = st.checkbox(
        "Include email headers (From / Subject / Reply-To, etc.) in analysis",
        value=False,
        help="If your email contains header lines at the top, they will be parsed and shown.",
    )

    email_text = st.text_area(
        "Email body",
        height=300,
        placeholder="Paste the full email message here...\n\nExample:\nSubject: Urgent - verify your account\nFrom: security@paypa1-security.com\nReply-To: attacker@evil.com\n\nDear customer, your account will be suspended. Click http://bit.ly/3x9fake to verify now.",
    )

    col_btn, col_clear = st.columns([1, 1])
    with col_btn:
        analyze = st.button("🔍 Analyze Email", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.rerun()

    if analyze:
        if not email_text or not email_text.strip():
            st.error("Please enter email text before analysing.")
        else:
            payload = load_model_payload(MODEL_PATH)
            if payload is None:
                st.error(
                    "No trained model found. Run `python train.py` first, "
                    "then return here."
                )
            else:
                headers = parse_headers(email_text) if include_headers else {}
                res = predict_email(email_text, payload)

                st.session_state.analysis_count += 1
                st.session_state.history.append({
                    "email": email_text[:200],
                    "label": res["label"],
                    "confidence": res["confidence"],
                    "urls": res["urls"],
                })

                # Result banner
                if res["is_phishing"]:
                    st.markdown(
                        f'<div style="padding:1rem; background:#7f1d1d; color:#fecaca; '
                        f'border-radius:8px; border:1px solid #991b1b;">'
                        f'<strong>🔴 Prediction: PHISHING</strong></div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="padding:1rem; background:#14532d; color:#bbf7d0; '
                        f'border-radius:8px; border:1px solid #166534;">'
                        f'<strong>🟢 Prediction: SAFE (Legitimate)</strong></div>',
                        unsafe_allow_html=True,
                    )

                c1, c2 = st.columns(2)
                with c1:
                    if res["confidence"] >= 0:
                        st.metric("Model confidence", f"{res['confidence']:.2%}")
                    else:
                        st.metric("Model confidence", "N/A")
                with c2:
                    if res["proba"] is not None:
                        st.metric("P(Phishing)", f"{res['proba'][1]:.2%}")
                        st.metric("P(Legitimate)", f"{res['proba'][0]:.2%}")

                st.caption(
                    "Confidence is the model's calibrated probability estimate — "
                    "a guide, not a guarantee."
                )

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

                # Headers
                if headers:
                    st.subheader("📧 Parsed headers")
                    for k, v in headers.items():
                        st.code(f"{k}: {v}")
                    reply_to = headers.get("reply-to", "").lower()
                    from_ = headers.get("from", "").lower()
                    if reply_to and from_ and reply_to not in from_:
                        st.warning("⚠️ Reply-To domain differs From domain — common in spoofing.")

                # Suspicious keywords
                st.subheader("🔎 Risk indicators detected")
                if res["reasons"]:
                    for r in res["reasons"]:
                        st.write(f"• {r}")
                else:
                    st.write("No high-risk hand-crafted indicators detected.")

                # URLs
                st.subheader("🔗 URLs extracted")
                if res["urls"]:
                    st.write(f"{len(res['urls'])} URL(s) found:")
                    for u in res["urls"]:
                        suspicious = (
                            any(d in u.lower() for d in ["bit.ly", "tinyurl"]) or
                            "@" in u or "%" in u or u.lower().startswith("http://")
                        )
                        tag = " ⚠️ suspicious" if suspicious else ""
                        st.code(u + tag)
                    st.caption("Links are shown for inspection only — never click them.")
                else:
                    st.write("No URLs found.")

                # Explanation
                st.subheader("💡 Why this classification?")
                if res["reasons"]:
                    for r in res["reasons"]:
                        st.write(f"• {r}")
                else:
                    st.write(
                        "No strong hand-crafted indicators; decision driven by the "
                        "TF-IDF text model."
                    )

                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.caption(
                    "False positives (safe flagged phishing) and false negatives "
                    "(phishing missed) can occur. Use human judgement alongside this tool."
                )

# ---------------------------------------------------------------------------
# Tab C: Batch Analyzer
# ---------------------------------------------------------------------------

elif page == "C. Batch Analyzer":
    st.title("📦 Batch Analyzer")
    st.subheader("Upload a CSV of emails for bulk scoring.")

    st.write(
        "CSV must contain a column named **`Email Text`**. "
        "Optionally include a **`label`** column (1=phishing, 0=safe) "
        "to compute batch evaluation metrics."
    )

    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Failed to read CSV: {exc}")
            batch_df = None

        if batch_df is not None:
            if "Email Text" not in batch_df.columns:
                st.error("CSV missing required column: `Email Text`")
            else:
                payload = load_model_payload(MODEL_PATH)
                if payload is None:
                    st.error("No trained model found. Run `python train.py` first.")
                else:
                    with st.spinner("Scoring emails..."):
                        results = []
                        for text in batch_df["Email Text"].astype(str).fillna(""):
                            r = predict_email(text, payload)
                            results.append(r)

                        out = batch_df.copy()
                        out["prediction"] = [r["label"] for r in results]
                        out["is_phishing"] = [r["is_phishing"] for r in results]
                        out["confidence"] = [r["confidence"] if r["confidence"] >= 0 else None for r in results]
                        out["p_phishing"] = [r["proba"][1] if r["proba"] is not None else None for r in results]

                        st.session_state.analysis_count += len(results)

                    st.success(f"Scored {len(results)} email(s).")
                    st.dataframe(
                        out[["Email Text", "prediction", "confidence", "p_phishing"]]
                        .head(50),
                        use_container_width=True,
                    )

                    # Download
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download scored CSV",
                        data=csv_bytes,
                        file_name="scored_emails.csv",
                        mime="text/csv",
                    )

                    # Batch metrics if labels present
                    if "label" in out.columns:
                        y_true = out["label"].astype(int).to_numpy()
                        y_pred = out["is_phishing"].astype(int).to_numpy()
                        from src.evaluation import evaluate_model_full
                        metrics = evaluate_model_full(y_true, y_pred)
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                        c2.metric("Precision", f"{metrics['precision']:.2%}")
                        c3.metric("Recall", f"{metrics['recall']:.2%}")
                        c4.metric("F1-score", f"{metrics['f1']:.2%}")

# ---------------------------------------------------------------------------
# Tab D: Model Performance
# ---------------------------------------------------------------------------

elif page == "D. Model Performance":
    st.title("📈 Model Performance")
    st.subheader("Evaluation on the held-out 20% test set")

    if df is None:
        st.error("Dataset not loaded.")
    elif not os.path.exists(MODEL_PATH):
        st.warning(
            "No trained model found. Train first:\n\n"
            "```bash\npython train.py\n```"
        )
    else:
        from src.preprocessing import split_train_test
        from src.evaluation import (
            classification_report_text, save_confusion_matrix,
            save_roc_pr_curves, save_model_comparison, save_feature_importance,
        )

        payload = load_model_payload(MODEL_PATH)
        model = payload["model"]
        model_name = payload.get("model_name", "Unknown")
        extractor = payload["extractor"]

        train_df, test_df = split_train_test(df, test_size=0.2, random_state=42)
        X_test = extractor.transform(test_df["Email Text"].tolist())
        y_test = test_df["label"].to_numpy()
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        from src.evaluation import evaluate_model_full
        metrics = evaluate_model_full(y_test, y_pred, y_proba)

        st.write(f"**Best model:** {model_name}")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        c2.metric("Precision", f"{metrics['precision']:.2%}")
        c3.metric("Recall", f"{metrics['recall']:.2%}")
        c4.metric("F1-score", f"{metrics['f1']:.2%}")
        if y_proba is not None:
            c5.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion matrix")
            cm_path = save_confusion_matrix(y_test, y_pred)
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
        with col2:
            st.subheader("ROC & PR curves")
            curves_path = save_roc_pr_curves(y_test, y_proba)
            if os.path.exists(curves_path):
                st.image(curves_path, use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Classification report")
        st.text(classification_report_text(y_test, y_pred))

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Model comparison (all algorithms)")
        from src.model_training import BASE_MODELS
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import VotingClassifier
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        X_train = extractor.fit_transform(train_df["Email Text"].tolist())
        y_train = train_df["label"].to_numpy()

        all_results = {}
        for name, base in BASE_MODELS.items():
            if name == "Linear SVM":
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            elif hasattr(base, "predict_proba"):
                clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
            else:
                clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            clf.fit(X_train, y_train)
            yp = clf.predict(X_test)
            all_results[name] = evaluate_model_full(y_test, yp)

        estimators = [(n, None) for n in BASE_MODELS]
        # Build voting ensemble from already-fitted calibrated models is tricky;
        # just show individual model comparison.
        comp_path = save_model_comparison(all_results)
        if os.path.exists(comp_path):
            st.image(comp_path, use_container_width=True)
        st.dataframe(
            pd.DataFrame(all_results).T.round(4),
            use_container_width=True,
        )

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Feature importance (permutation)")
        try:
            fi_path = save_feature_importance(model, extractor, X_test, y_test)
            if fi_path and os.path.exists(fi_path):
                st.image(fi_path, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance unavailable: {exc}")

        st.warning(
            "⚠️ No model achieves 100% accuracy. False positives inconvenience "
            "users; false negatives are a security risk. Use as one layer of defence."
        )

# ---------------------------------------------------------------------------
# Tab E: Dataset Information
# ---------------------------------------------------------------------------

elif page == "E. Dataset Information":
    st.title("📁 Dataset Information")
    st.subheader("About the data used to train the model")

    if df is None:
        st.error("Dataset not loaded.")
    else:
        total = len(df)
        phishing = int((df["label"] == 1).sum())
        safe = int((df["label"] == 0).sum())

        st.write("**Dataset name:** Phishing Email Detection Dataset")
        st.write(
            "**Source:** Public GitHub repository "
            "(Foxriver-2005 / Real-Time-Email-Phishing-Detection, "
            "`Datasets/Phishing_Email_2.csv`)"
        )
        st.write("**Format:** CSV — `Email Text`, `label` (1=phishing, 0=safe), `Email Type`")
        st.write("**Missing values:** 0 (rows with missing text removed)")
        st.write("**Duplicate records:** 0 (removed)")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total records", f"{total:,}")
        c2.metric("Legitimate (Safe)", f"{safe:,}")
        c3.metric("Phishing", f"{phishing:,}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Train / test split (stratified 80/20)")
        train_df, test_df = split_train_test(df, test_size=0.2, random_state=42)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train samples", f"{len(train_df):,}")
        c2.metric("Test samples", f"{len(test_df):,}")
        c3.metric("Train phishing", f"{int((train_df['label']==1).sum()):,}")
        c4.metric("Test phishing", f"{int((test_df['label']==1).sum()):,}")

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Sample emails")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Legitimate (safe) example")
            safe_sample = df[df["label"] == 0]["Email Text"].iloc[0]
            st.text_area("Safe email", value=safe_sample[:1000], height=220, disabled=True)
        with col2:
            st.caption("Phishing example")
            phish_sample = df[df["label"] == 1]["Email Text"].iloc[0]
            st.text_area("Phishing email", value=phish_sample[:1000], height=220, disabled=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.subheader("Preview (first 10 rows)")
        st.dataframe(df[["Email Text", "Email Type"]].head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "Phishing Email Detection System — Python, scikit-learn, pandas, NLTK, "
    "Matplotlib, Seaborn, Streamlit, Joblib. For educational / academic use. "
    "Not a guarantee of email safety."
)
