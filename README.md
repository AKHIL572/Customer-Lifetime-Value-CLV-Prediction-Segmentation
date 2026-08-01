# Customer Lifetime Value (CLV) Prediction — Online Retail

Predicting 6-month forward customer revenue from behavioral purchase history, built as a full end-to-end data science pipeline: leakage-safe modeling, a modular tested codebase, an interactive Streamlit app, and a Power BI dashboard — all kept consistent through a single shared configuration.

> **Honest disclosure up front:** the trained model does not currently outperform a naive "past spend = future spend" baseline (£899.76 MAE vs. £794.10 MAE). This is documented throughout the project rather than hidden — see [Model Performance](#model-performance) below.

---

## Table of Contents
- [Overview](#overview)
- [Key Results](#key-results)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [Data](#data)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Streamlit App](#streamlit-app)
- [Power BI Dashboard](#power-bi-dashboard)
- [Testing](#testing)
- [Known Limitations & Future Work](#known-limitations--future-work)
- [License](#license)

---

## Overview

Online retailers acquire and retain customers at a cost, but not all customers are equally valuable going forward. This project predicts each customer's revenue over the **next 6 months** using only their purchase behavior *before* a cutoff date — with the target window explicitly bounded to prevent data leakage — so marketing and CRM teams can prioritize retention spend toward customers who are actually likely to be valuable, not just customers who were valuable historically.

Built on the "Online Retail" dataset: ~542K transactions from a UK-based online gift retailer, December 2010 – December 2011.

---

## Key Results

| Finding | Value |
|---|---|
| Revenue from top 20% of customers | ~74.6% |
| One-time buyers (% of customers) | 34.7% |
| Revenue from one-time buyers | ~7.2% |
| Customer segments identified (K-Means on RFM) | Champions, Loyal Mid-Value, New/Low Engagement, At Risk/Lapsed |
| Champion model | Gradient Boosting Regressor |
| Test MAE | £899.76 |
| Test R² | 0.381 |
| Naive baseline MAE | £794.10 |
| Beats naive baseline? | **No** (-13.3%) — disclosed openly, see below |

---

## Project Structure

```
ONLINE_RETAIL/
│
├── Dataset/
│   ├── OnlineRetail.csv                          # raw data
│   ├── cleaned_online_retail_transactions.csv    # cleaned, audit-tracked
│   ├── cancelled_orders.csv                      # cancellations, preserved separately
│   ├── data_cleaning_audit_log.csv               # before/after row counts per cleaning step
│   ├── customer_level_eda_summary.csv            # full-history RFM + K-Means segments
│   ├── segment_profile_summary.csv               # segment-level profile table
│   └── clv_predictions.csv                       # per-customer model output + segment
│
├── Models/
│   ├── clv_model.pkl
│   ├── clv_features.pkl
│   ├── segment_thresholds.json                   # single source of truth for segment cutoffs
│   └── model_card.json                           # training metadata, metrics, limitations
│
├── Notebook/
│   ├── 1_data_understanding.ipynb                # cleaning, with full audit trail
│   ├── 2_eda.ipynb                                # EDA + K-Means segmentation
│   └── 3_preprocessing_&_modeling.ipynb           # leakage-safe modeling, baseline-checked
│
├── src/
│   ├── config.py                                  # shared paths, features, thresholds
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── tests/
│   └── test_pipeline.py                           # unit tests incl. leakage-bug regression test
│
├── Power_BI_dashboard/
│   └── power_BI_Dashboard.pbix                    # 3 pages
│
├── app.py                                          # Streamlit app, 4 tabs
└── requirements.txt
```

---

## Tech Stack

**Language:** Python 3.12
**Data & ML:** pandas, numpy, scikit-learn, XGBoost, joblib
**Visualization:** matplotlib, seaborn
**App:** Streamlit
**BI:** Power BI (DAX, Power Query)
**Testing:** pytest
**Environment:** Jupyter Notebook

---

## Getting Started

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd ONLINE_RETAIL
pip install -r requirements.txt
```

### 2. Run the notebooks in order (optional — pre-trained artifacts are already included)

```bash
jupyter notebook Notebook/1_data_understanding.ipynb
jupyter notebook Notebook/2_eda.ipynb
jupyter notebook "Notebook/3_preprocessing_&_modeling.ipynb"
```

### 3. Or retrain from the command line

```bash
python -m src.train
```

This regenerates `Models/clv_model.pkl`, `clv_features.pkl`, `segment_thresholds.json`, and `model_card.json`.

### 4. Run the test suite

```bash
python -m pytest tests/ -v
```

### 5. Launch the Streamlit app

```bash
streamlit run app.py
```

---

## Data

- **Source:** "Online Retail" dataset (UCI Machine Learning Repository)
- **Raw:** 541,909 transaction rows, Dec 2010 – Dec 2011
- **Cleaned:** 391,150 rows, 0 missing values, 0 duplicates — every removal decision is documented and quantified in `data_cleaning_audit_log.csv`
- **Customers (full history):** 4,334
- **Customers in modeling population** (active before the cutoff date): 2,791
- **Currency:** GBP (£) throughout

---

## Methodology

1. **Data cleaning** — investigate before acting; every decision documented; cancellations preserved separately rather than discarded
2. **EDA** — quantified business patterns (Pareto concentration, one-time vs. repeat buyers) plus unsupervised K-Means segmentation, used later as an independent sanity check on the model
3. **Leakage-safe feature engineering** — a single cutoff date strictly separates the feature window from an explicitly bounded target window
4. **Baseline-first evaluation** — a naive "past spend = future spend" baseline is computed before any model is judged
5. **Evidence-based model selection** — 5 candidate models compared via cross-validation; champion chosen on results, not assumption
6. **Hyperparameter tuning** — `RandomizedSearchCV` on the selected champion only
7. **Business validation** — model predictions cross-checked against the independently-derived K-Means segments
8. **Deployment** — same trained artifacts served through both the Streamlit app and the Power BI dashboard

---

## Model Performance

| Metric | Value |
|---|---|
| Model | Gradient Boosting Regressor |
| Test MAE | £899.76 (58.7% of mean CLV) |
| Test RMSE | £2,767.74 |
| Test R² | 0.381 |
| Naive baseline MAE | £794.10 |
| Improvement over baseline | **-13.3%** |
| % customers with £0 six-month CLV | 30.5% |

The model does not currently beat a simple "assume next 6 months looks like historical spend" heuristic. This is disclosed in `Models/model_card.json`, the Streamlit app's **Model Info** tab, and the Power BI dashboard's **Customer Value** page — not hidden. Despite this, the model's predictions correctly rank the independently-derived customer segments in the expected order (Champions highest, At Risk/Lapsed lowest), indicating it has learned real behavioral signal even where it hasn't yet beaten the baseline on raw error.

See [Known Limitations](#known-limitations--future-work) for what would need to change to close this gap.

---

## Streamlit App

Four tabs, all backed by the same trained model and segment thresholds used everywhere else in the project:

- **🔍 Customer Lookup** — select a real `CustomerID` and view their actual behavioral profile and predicted CLV
- **🧪 What-If Scenario** — manually enter a hypothetical customer's behavior; derived fields (e.g. Avg Order Value) are auto-computed, not manually entered, to prevent inconsistent inputs
- **📤 Batch Upload** — upload a CSV of customers and download scored results
- **ℹ️ Model Info** — transparent model card: metrics, baseline comparison, and known limitations, shown directly to the end user

Run locally with `streamlit run app.py`. *(Not currently deployed to a public URL — add your deployment link here if published.)*

---

## Power BI Dashboard

3 pages, all built on the cleaned/audited data:

1. **Executive Sales Dashboard** — orders, revenue, quantity, AOV, revenue trend, order status, top countries, revenue map, peak ordering hours
2. **Product & Customer Intelligence** — customer/order KPIs, weekly sales pattern, top products, product revenue-vs-quantity, customer detail table (bridges into `Predicted_CLV` and `CLV_Segment`)
3. **Customer Value & CLV** — Total/Average Predicted CLV, segment breakdown, Frequency-vs-Monetary scatter by segment, Top 20 customers by predicted CLV, and a Model Performance & Limitations panel surfacing the honest baseline comparison to business users

File: `Power_BI_dashboard/power_BI_Dashboard.pbix`

---

## Testing

```bash
python -m pytest tests/ -v
```

6 unit tests, including a regression test that specifically catches the original target-window leakage bug (an early version of the target definition had no upper bound and only produced correct results by coincidence — this is now explicitly tested against).

---

## Known Limitations & Future Work

- **Model does not yet beat its baseline** — next steps: richer features (product-category diversity, return behavior), or more historical data to enable proper multi-period walk-forward validation
- **No true walk-forward validation** — dataset length (~12.3 months) only supports one non-overlapping 6-month split
- **Cold-start customers unsupported** — new customers with no purchase history cannot be scored by this model; would need a separate rule-based fallback in production
- **UK-dominant data** — performance on other countries is lower-confidence
- **`clv_predictions.csv` is a training-run snapshot** — retraining via `train.py` does not currently regenerate it automatically; needs to be re-exported if the model changes

---

## 👤 Author

**Akhil T V**  
Aspiring Data Scientist | Data Analyst  
LinkedIn: https://www.linkedin.com/in/akhil-t-v/

---

⭐ If you found this project useful, feel free to star the repository!
