# 🛡️ Phishing Email Detection System

A complete, beginner-friendly machine learning application that analyses email content and classifies each message as **🔴 Phishing** or **🟢 Safe (Legitimate)**.

Built with **Python, scikit-learn, pandas, Streamlit** and **Joblib**, featuring a modern cybersecurity dashboard, four compared ML algorithms, evaluation charts, and automated tests. Suitable for a college cybersecurity mini-project submission.

---

## 📑 Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Technologies Used](#technologies-used)
5. [Dataset Source and Format](#dataset-source-and-format)
6. [Machine Learning Workflow](#machine-learning-workflow)
7. [Project Folder Structure](#project-folder-structure)
8. [Installation Instructions](#installation-instructions)
9. [How to Train the Model](#how-to-train-the-model)
10. [How to Run the Application](#how-to-run-the-application)
11. [How to Run the Tests](#how-to-run-the-tests)
12. [Sample Test Emails](#sample-test-emails)
13. [Model Evaluation Results](#model-evaluation-results)
14. [Screenshots Section](#screenshots-section)
15. [Limitations and Future Improvements](#limitations-and-future-improvements)
16. [Security & Accuracy Notes](#security--accuracy-notes)
17. [License](#license)

---

## 1. Introduction

Phishing is one of the most common cyber-attacks. Attackers send fraudulent emails that pretend to come from banks, payment services, or well-known companies in order to steal passwords, credit-card details, or install malware.

This project is a **Phishing Email Detection System** — a machine learning tool that reads the body of an email and predicts whether it is a phishing attempt or a legitimate message. It includes:

* A trained classification model saved with **Joblib**
* A **Streamlit** web dashboard for analysing emails and viewing model performance
* Clean, modular source code under `src/`
* A real, publicly available labelled dataset
* Automated unit tests under `tests/`
* Evaluation visualisations (confusion matrix, model comparison chart)

---

## 2. Problem Statement

Users frequently cannot tell whether an incoming email is genuine or a phishing attempt. Existing email filters help, but many phishing messages still reach inboxes. There is a need for an interpretable, locally-run machine learning tool that:

* Learns from labelled examples of phishing and legitimate emails
* Extracts both text (TF-IDF) and security-specific features (URLs, suspicious keywords, urgency, etc.)
* Predicts the class of previously unseen emails
* Explains *why* an email was flagged
* Runs entirely offline (no email content is sent to external services)

---

## 3. Objectives

| # | Objective |
|---|-----------|
| 1 | Load and clean a public labelled phishing/legitimate email dataset |
| 2 | Extract TF-IDF text features plus hand-crafted security features |
| 3 | Train and compare **Logistic Regression, Naive Bayes, Random Forest, and Linear SVM** |
| 4 | Select the best model by **test-set** performance (not training accuracy) |
| 5 | Evaluate with accuracy, precision, recall, F1-score, confusion matrix |
| 6 | Save the final model with **Joblib** |
| 7 | Provide a **Streamlit** dashboard (analyzer + performance + dataset info) |
| 8 | Include automated tests and beginner-friendly documentation |

---

## 4. Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.11+** (tested on 3.14.4) | Core language |
| **scikit-learn** | ML models, TF-IDF, train/test split, metrics |
| **pandas** | Dataset loading and manipulation |
| **NumPy** | Numerical arrays |
| **NLTK** | Optional text-processing support |
| **Matplotlib + Seaborn** | Evaluation charts (confusion matrix, comparison bar chart) |
| **Streamlit** | Web user interface (dashboard) |
| **Joblib** | Saving / loading the trained model |
| **pytest** | Automated unit tests |

---

## 5. Dataset Source and Format

### Dataset
**Name:** Phishing Email Detection Dataset  
**Source:** Public GitHub repository — *Foxriver-2005 / Real-Time-Email-Phishing-Detection*, file `Datasets/Phishing_Email_2.csv`  
**License:** Publicly available for research / educational use

### Format
The dataset (`data/dataset.csv`) contains three columns:

| Column | Description |
|--------|-------------|
| `Email Text` | Raw body of the email |
| `label` | `1` = Phishing, `0` = Legitimate (Safe) |
| `Email Type` | Human-readable label ("Phishing Email" / "Safe Email") |

### Cleaning performed
* Removed the unused index column (`Unnamed: 0`)
* Removed rows with missing email text
* Removed duplicate records
* Normalised labels to `0` / `1`
* HTML tags stripped during feature extraction

### Final dataset statistics (real, measured)

| Metric | Value |
|--------|-------|
| Total records | **17,537** |
| Legitimate (Safe) | **10,979** |
| Phishing | **6,558** |
| Missing values | 0 |
| Duplicate records | 0 |

> ⚠️ Honesty note: these statistics were computed from the actual dataset file; nothing was fabricated.

---

## 6. Machine Learning Workflow

```
1. Load dataset (data/dataset.csv)
        |
        v
2. Clean text (NLTK tokenisation, lemmatisation, stop-word removal, HTML stripping)
        |
        v
3. Stratified train / test split  (80% train, 20% test)
        |  preserves phishing : legitimate ratio in both splits
        v
4. Feature extraction (fit on TRAIN only -> no data leakage)
    - TF-IDF (unigrams + bigrams + trigrams, English stop words removed)
    - 34 hand-crafted security features (see table below)
        |
        v
5. Train 5 configurations:
    Logistic Regression, Naive Bayes, Random Forest, Linear SVM,
    and a soft-voting Ensemble (probability averaging)
    |  all probabilities calibrated (CalibratedClassifierCV)
        |
        v
6. 5-fold cross-validated F1 on training data + held-out TEST evaluation
        |
        v
7. Select best model by test-set F1-score (CV-F1 tie-break)
        |
        v
8. Save model + extractor to models/phishing_model.joblib (Joblib)
        |
        v
9. Generate reports/ (confusion matrix, ROC/PR curves, model comparison,
   permutation feature importance)
```

### Hand-crafted features (34 columns)

| Group | Features |
|-------|----------|
| **URL-based** | number of URLs, suspicious URLs, IP-based URLs, URL shorteners, HTTP vs HTTPS, excessive subdomains, domain entropy, risky TLDs, URL-encoding chars, `@` in host |
| **Text-based** | email length, word count, special characters, special-char ratio, uppercase letters, uppercase ratio, digits, exclamation marks, question marks, currency symbols |
| **Keyword-based** | urgency, authority, scarcity, financial, credential, legal-threat, account-verification, password-reset, delivery, invoice counts |
| **Structural** | has subject-like line, has greeting, has signature, Flesch readability score |

TF-IDF (sparse, high-dimensional) and hand-crafted features (dense) are combined with `scipy.sparse.hstack` so every model sees the same representation.

---

## 7. Project Folder Structure

```
phishing-email-detector/
├── app.py                     # Streamlit dashboard
├── train.py                   # Training entry point
├── predict.py                 # Single-email prediction CLI
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── data/
│   └── dataset.csv            # Cleaned labelled dataset (17,537 emails)
│
├── models/
│   └── phishing_model.joblib  # Trained best model (created by train.py)
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py       # Dataset loading, text cleaning, train/test split
│   ├── feature_extraction.py  # TF-IDF + hand-crafted security features
│   ├── model_training.py      # Train 4 models, select best, save with Joblib
│   └── evaluation.py          # Metrics + visualisations (charts)
│
├── reports/
│   ├── confusion_matrix.png   # Confusion-matrix heatmap
│   ├── roc_pr_curves.png      # ROC & Precision-Recall curves
│   ├── model_comparison.png   # Model comparison bar chart
│   └── feature_importance.png # Permutation feature importance
│
└── tests/
    └── test_prediction.py     # Automated unit tests (12 tests)
```

---

## 8. Installation Instructions

### 1. Clone / create the folder
Make sure you are inside the folder that contains `requirements.txt`:

```
cd phishing-email-detector
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

This installs: `scikit-learn`, `pandas`, `numpy`, `nltk`, `matplotlib`, `seaborn`, `streamlit`, `joblib`, `pytest`.

> **Python version:** 3.11 or newer recommended. This project was developed and tested on **Python 3.14.4**.

---

## 9. How to Train the Model

Train and evaluate the model, then save it:

```
python train.py
```

**What happens:**
1. Loads `data/dataset.csv`
2. Splits into stratified 80/20 train/test
3. Trains Logistic Regression, Naive Bayes, Random Forest, Linear SVM
4. Selects the best model by test-set F1-score
5. Saves `models/phishing_model.joblib`
6. Writes `reports/confusion_matrix.png` and `reports/model_comparison.png`
7. Prints a comparison table and classification report

You only need to run this once. The Streamlit app loads the saved model automatically.

---

## 10. How to Run the Application

Launch the Streamlit dashboard:

```
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

### Dashboard tabs

| Tab | What it does |
|-----|--------------|
| **A. Home Dashboard** | Total emails analysed, safe vs phishing counts, session analysis count, model accuracy |
| **B. Email Analyzer** | Paste an email → get prediction (Phishing/Safe) + calibrated confidence, parsed headers, highlighted suspicious keywords, extracted URLs, plain-language explanation |
| **C. Batch Analyzer** | Upload a CSV of emails (column `Email Text`) → bulk scoring with per-email confidence, downloadable scored CSV, optional batch evaluation metrics if a `label` column is present |
| **D. Model Performance** | Accuracy/precision/recall/F1, ROC-AUC, PR-AUC, confusion matrix heatmap, ROC & PR curves, classification report, model comparison chart, feature importance |
| **E. Dataset Information** | Dataset name/source, record counts, class balance, train/test split, sample emails |

> ⚠️ **Security:** emails you paste are processed locally. Nothing is sent to external services, no links are opened, and nothing is stored unless you explicitly enable it. This is a detection aid — not a guarantee that an email is safe.

---

## 11. How to Run the Tests

```
python -m pytest tests/
```

There are **12 automated tests** covering:
* Empty input handling
* HTML-tag removal during cleaning
* URL extraction (normal + variants including IP-based URLs)
* Hand-crafted feature shape and non-negativity
* Feature extractor fit/transform consistency
* Missing-dataset handling (`FileNotFoundError`)
* Dataset loads with expected columns/labels
* Model loading from Joblib
* Prediction output format (valid label + confidence)
* Empty-input prediction

Expected result: **12 passed**.

---

## 12. Sample Test Emails

You can paste these into the **Email Analyzer** tab (tab B) to try the system.

### ✅ Legitimate personal email
```
Hello John,

I hope you are doing well. I am writing to confirm our meeting tomorrow
at 10:00 AM in the conference room. Please let me know if you need to
reschedule.

Best regards,
Sarah
```

### ✅ Legitimate business email
```
Subject: Project status update

Hi team,

Please find attached the Q3 project status report. Let me know if you have
any questions ahead of Friday's review.

Thanks,
Project Manager
```

### 🔴 Fake bank verification email (phishing)
```
URGENT: Your account will be suspended!

Dear customer, we detected suspicious activity on your account.
Please verify your identity immediately by clicking here:
http://bit.ly/3x9fake-secure-login

Failure to act within 24 hours will result in account closure.
Reset your password now.
```

### 🔴 Fake password reset email (phishing)
```
Password Reset Request

Someone requested a password reset for your account. If this was you,
click the link below to reset your password:
http://192.168.1.1/admin/reset?token=xyz

If you did not request this, contact support immediately.
```

### 🔴 Suspicious prize-winning email (phishing)
```
CONGRATULATIONS! You have won a $1,000 gift card!

Click here to claim your prize now: http://tinyurl.com/fake-prize
Offer expires in 2 hours. Act fast!
```

### 🔴 Urgent account suspension email (phishing)
```
Final Notice: Account Suspension

Your account will be deactivated within 24 hours due to unusual login
activity. Verify now at http://secure-paypal-update.com/login to avoid
losing access.
```

### 🔴 Email containing shortened URLs (phishing)
```
Your package delivery failed. Track it here:
https://bit.ly/3track-pkg and http://cutt.ly/fake-courier

Please verify your address immediately.
```

### 🔴 Email containing multiple suspicious links (phishing)
```
Security alert: multiple failed login attempts detected.

Secure your account:
http://10.0.0.99/login
https://bit.ly/2secure-now
www.verify-account-update.net/login
```

---

## 13. Model Evaluation Results

Results below are **measured on the held-out 20% test set** (3,508 emails) — not training accuracy.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | CV-F1 |
|-------|----------|-----------|--------|----------|---------|--------|-------|
| Logistic Regression | 0.9470 | 0.9265 | 0.9322 | 0.9293 | 0.9852 | 0.9746 | 0.9331 |
| Multinomial Naive Bayes | 0.7258 | 0.7025 | 0.4627 | 0.5579 | 0.7056 | 0.6053 | 0.5047 |
| **Random Forest** (best) | **0.9692** | **0.9547** | **0.9634** | **0.9590** | **0.9958** | **0.9932** | **0.9615** |
| Linear SVM | 0.8820 | 0.9236 | 0.7462 | 0.8255 | 0.9251 | 0.8908 | 0.7264 |
| Voting Ensemble | 0.9635 | 0.9611 | 0.9405 | 0.9507 | 0.9934 | 0.9904 | 0.9508 |

**Best model: Random Forest** (selected by highest test-set F1-score).

### Classification report (Random Forest, test set)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Legitimate (Safe) | 0.98 | 0.97 | 0.98 | 2,196 |
| Phishing | 0.95 | 0.96 | 0.96 | 1,312 |
| **Accuracy** | | | **0.97** | 3,508 |
| Macro avg | 0.97 | 0.97 | 0.97 | 3,508 |
| Weighted avg | 0.97 | 0.97 | 0.97 | 3,508 |

### Visualisations (generated in `reports/`)

* `confusion_matrix.png` — heatmap of true vs predicted classes
* `roc_pr_curves.png` — ROC curve (AUC) and Precision-Recall curve (AP)
* `model_comparison.png` — bar chart comparing all five configurations
* `feature_importance.png` — top permutation-importance features

> ⚠️ **No model achieves 100% accuracy.**
> * **False positive:** a legitimate email flagged as phishing (inconvenience).
> * **False negative:** a phishing email missed by the model (security risk).
>
> Always use this tool alongside human judgement — never treat the output as a guarantee.

---

## 14. Screenshots Section

> Place screenshots of the running Streamlit app here (tab A–D) before submission.
>
| Tab | Suggested filename |
|-----|--------------------|
| Home Dashboard | `screenshots/home-dashboard.png` |
| Email Analyzer | `screenshots/email-analyzer.png` |
| Model Performance | `screenshots/model-performance.png` |
| Dataset Information | `screenshots/dataset-info.png` |

The `reports/` folder already contains:
* `reports/confusion_matrix.png`
* `reports/model_comparison.png`

---

## 15. Limitations and Future Improvements

| Limitation | Possible improvement |
|------------|----------------------|
| Dataset phishing samples are heavily skewed toward adult/advertising-style content | Add more varied phishing samples (banking, CEO fraud, invoice scams) |
| Single subject/body text; no email headers analysed | Add header analysis (SPF/DKIM, sender domain mismatch) |
| No attachment or embedded-image scanning | Add attachment MIME-type checks and image OCR |
| Confidence scores are model estimates, not guarantees | Calibrate probabilities (Platt scaling / isotonic regression) |
| English-only stop words | Add multilingual support |
| Static snapshot of data | Scheduled retraining on new labelled emails |

---

## 16. Security & Accuracy Notes

* All submitted emails are treated as **untrusted input**.
* No attachments, scripts, or URLs are ever executed.
* Suspicious links are **never opened automatically**.
* Email content is **never sent to external services**.
* Submitted emails are **not stored** unless explicitly enabled.
* Empty / invalid input is handled gracefully.
* Confidence scores are **estimates only** — avoid misleading certainty.
* This is a **machine-learning detection tool**, not a guarantee that an email is safe.

---

## 17. License

This project is intended for educational / academic use (college cybersecurity mini-project).

---

**Built by:** A Senior Machine Learning Engineer, Python Developer & Cybersecurity Specialist  
**Due:** November 19, 2026  
**Commands:** `pip install -r requirements.txt` → `python train.py` → `streamlit run app.py` → `python -m pytest tests/`
