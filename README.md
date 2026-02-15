# 🛍️ Customer Lifetime Value (CLV) Prediction & Business Intelligence System

## 📌 Project Overview

This project builds an end-to-end **Customer Lifetime Value (CLV) prediction system** using machine learning and translates the predictions into actionable business insights through an interactive Power BI dashboard and a deployed Streamlit web application.

The objective is to help businesses:

- Identify high-value customers
- Detect churn risk
- Optimize retention strategies
- Improve revenue forecasting
- Support data-driven decision making

This is a full-stack data science project including:

- Data cleaning & preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering (RFM-based)
- Machine learning modeling & tuning
- Model deployment (Streamlit)
- Business intelligence dashboard (Power BI)

---

# 🏗️ Project Architecture

```
ONLINE_RETAILS/
│
├── Dataset/                    # Raw & processed datasets
├── Models/                     # Saved ML models & scalers
├── Notebook/                   # Research & experimentation notebooks
│   ├── 1_data_understanding.ipynb
│   ├── 2_eda.ipynb
│   └── 3_preprocessing_&_modeling.ipynb
│
├── src/                        # Production-level modular code
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── train.py
│   └── predict.py
│
├── Power_BI_dashboard/         # Dashboard screenshots / .pbix file
├── app.py                      # Streamlit deployment app
├── requirements.txt            # Dependencies
└── README.md
```

---

# 📊 Dataset Description

Dataset: **Online Retail Dataset**

Contains transactional data including:

- InvoiceNo
- StockCode
- Description
- Quantity
- InvoiceDate
- UnitPrice
- CustomerID
- Country

The dataset represents real-world e-commerce transactions.

---

# 🔍 1️⃣ Data Cleaning & Preparation

Performed in `1_data_understanding.ipynb`

### Steps:
- Removed duplicate rows
- Removed cancelled transactions
- Converted InvoiceDate to datetime
- Removed negative quantities
- Handled missing CustomerID values
- Created TotalPrice feature
- Saved cleaned dataset

---

# 📈 2️⃣ Exploratory Data Analysis (EDA)

Performed in `2_eda.ipynb`

### Key Analysis:
- Revenue distribution
- Customer purchase frequency
- Recency distribution
- Monetary value distribution
- Country-wise revenue contribution
- Pareto (80/20) revenue analysis
- Outlier detection
- Correlation matrix (RFM features)

### Business Insight:
- Small percentage of customers drive majority of revenue
- CLV is highly right-skewed
- Strong relationship between Frequency and Monetary value

---

# 🧠 3️⃣ Feature Engineering

Implemented in:

```
src/feature_engineering.py
```

### RFM Features Created:

- **Recency** → Days since last purchase
- **Frequency** → Number of unique transactions
- **Monetary** → Total spending
- **Average Order Value**
- **Customer Age**
- **CLV (6-month prediction window)**

---

# 🤖 4️⃣ Machine Learning Modeling

Performed in:

```
3_preprocessing_&_modeling.ipynb
src/train.py
```

---

## 🎯 Target Variable

```
CLV_6M
```

Predicted future 6-month revenue per customer.

---

## ⚙️ Preprocessing Steps

- Log transformation of skewed variables
- Train-test split (time-aware)
- Feature scaling using StandardScaler

---

## 🏆 Models Compared

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

---

## 📊 Model Evaluation Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Cross-validation

---

## 🥇 Final Model Selected

**Gradient Boosting Regressor**

Reason:
- Handles nonlinear relationships
- Performs well on tabular data
- Strong generalization performance

---

## 💾 Model Artifacts Saved

Stored inside:

```
Models/
```

Includes:
- Trained model (.pkl)
- Scaler
- Feature list

---

# 🚀 5️⃣ Model Deployment (Streamlit App)

File:

```
app.py
```

### Features:

- User inputs:
  - Recency
  - Frequency
  - Monetary
  - Avg Order Value
  - Customer Age
- Predicts CLV instantly
- Automatically assigns CLV segment:
  - High
  - Medium
  - Low
- Clean UI for business users

---

## ▶️ Run the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

# 📊 6️⃣ Power BI Dashboard

The Power BI dashboard transforms ML predictions into business strategy.

---

## 📄 Page 1 – Customer Segment Analysis

### Visuals:
- KPI Cards (Total CLV, Avg CLV, High CLV %)
- Bar chart (Revenue by CLV Segment)
- Donut chart (Customer Distribution)
- Scatter plot (Recency vs CLV)

---

## 📄 Page 2 – RFM Behavioral Insights

### Visuals:
- Recency Distribution
- Frequency vs Monetary Scatter
- Country-wise Revenue
- Revenue Concentration (Pareto)

---

## 📄 Page 3 – Strategic Business Insights

### Visuals:
- High-value customer risk analysis
- Revenue contribution by segment
- At-risk high CLV customers table
- Cumulative CLV curve (80/20 principle)

---

# 📈 Key Business Insights

- Top customers contribute disproportionately to total revenue
- High CLV customers with high recency indicate churn risk
- Retention campaigns should prioritize high-value segments
- Revenue concentration confirms Pareto principle

---

# 🧪 Reproducibility

All dependencies are listed in:

```
requirements.txt
```

Install using:

```bash
pip install -r requirements.txt
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Power BI
- Joblib

---

# 📌 Project Highlights

✔ End-to-end ML pipeline  
✔ Modular production-ready code structure  
✔ Hyperparameter tuning  
✔ Cross-validation  
✔ Business intelligence dashboard  
✔ Deployable web app  
✔ Real-world customer analytics use case  

---

# 🎯 Business Impact

This system enables companies to:

- Prioritize high-value customers
- Predict churn risk
- Improve retention ROI
- Optimize marketing spend
- Forecast future revenue

---

# 👤 Author

Akhil  
Aspiring Data Scientist | Machine Learning Enthusiast  

---

# 📬 Contact

Feel free to connect for collaboration or discussion.
akhilthottekkat135@gmail.com

---

# ⭐ If You Found This Useful

Please consider giving the repository a star ⭐
