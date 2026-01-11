# 📊 Customer Lifetime Value (CLV) Prediction & Analysis

Predicting customer lifetime value and analyzing customer behavior using machine learning, interactive dashboards, and business intelligence tools.

---

## 🚀 Project Overview

Customer Lifetime Value (CLV) is a key metric that estimates the total revenue a business can expect from a customer over a specific period.  
This project builds an **end-to-end CLV analytics system** that includes:

- Data cleaning and exploratory data analysis (EDA)
- RFM-based customer behavior analysis
- Machine learning model for 6-month CLV prediction
- Customer segmentation (Low / Medium / High value)
- Interactive Power BI dashboards
- Streamlit web app for real-time CLV prediction

---

## 🎯 Objectives

- Understand customer purchasing behavior
- Predict future customer value using historical data
- Segment customers based on predicted CLV
- Provide actionable insights through dashboards
- Deploy a simple web app for CLV prediction

---

## 🧠 Methodology

1. **Data Preprocessing**
   - Removed invalid transactions
   - Created TotalAmount feature
   - Handled missing values

2. **Exploratory Data Analysis**
   - Distribution of Quantity, UnitPrice, and Revenue
   - Customer-level aggregation
   - Outlier detection using IQR

3. **Feature Engineering**
   - Recency (days since last purchase)
   - Frequency (number of purchases)
   - Monetary value (total spend)
   - Total Quantity purchased

4. **Model Building**
   - Regression-based ML model for CLV prediction
   - Model trained on engineered customer features
   - Saved model and feature list for deployment

5. **Customer Segmentation**
   - Low, Medium, High CLV segments
   - Cumulative CLV and Pareto (80/20) analysis

6. **Visualization & Deployment**
   - Power BI dashboards for business insights
   - Streamlit app for interactive CLV prediction

---

## 🖥️ Power BI Dashboards

### 📄 Page 1 – CLV Overview
- Total Customers
- Average CLV
- Total Predicted CLV
- High CLV %
- CLV distribution by segment
- Cumulative CLV (Pareto curve)
- Filters for Recency and CLV Segment

### 📄 Page 2 – Customer Segment Analysis
- Customer count per CLV segment
- Average CLV by segment
- Revenue contribution per segment
- Average Recency, Frequency, and Monetary value
- Segment-wise behavioral insights

### 📄 Page 3 – CLV Distribution & Insights
- CLV rank-based cumulative analysis
- Revenue concentration patterns
- Customer contribution analysis

---

## 🌐 Streamlit Web Application

**Features:**
- User inputs:
  - Recency
  - Frequency
  - Monetary Value
  - Total Quantity
- Predicts 6-month CLV instantly
- Displays customer segment (Low / Medium / High)
- Simple, clean, business-friendly UI

---

## 📂 Project Structure

```
online_retails/
│
├── datasets/
│ └── cleaned_online_retail_transactions.csv
│
├── notebooks/
│ ├── 01_data_cleaning.ipynb
│ ├── 02_eda.ipynb
│ └── 03_feature_engineering_model.ipynb
│
├── models/
│ ├── clv_model.pkl
│ └── clv_features.pkl
│
├── streamlit_app/
│ ├── app.py
│ └── requirements.txt
│
├── powerbi/
│ └── CLV_Dashboard.pbix
│
└── README.md

```


---

## 🛠️ Technologies Used

- **Programming:** Python  
- **Data Analysis:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Dashboarding:** Power BI, DAX  
- **Web App:** Streamlit  
- **Version Control:** Git, GitHub  

---

## 📈 Key Insights

- A small percentage of customers contribute the majority of revenue
- High CLV customers have higher frequency and lower recency
- CLV segmentation helps target retention and marketing strategies
- Pareto analysis validates the 80/20 revenue rule

---

## 📌 Future Improvements

- Add churn prediction
- Extend CLV horizon (12 months)
- Include time-series models
- Deploy Streamlit app to cloud

---

## 👤 Author

**Akhil T V**  
Aspiring Data Scientist | Data Analyst  
LinkedIn: https://www.linkedin.com/in/akhil-t-v/

---

⭐ If you found this project useful, feel free to star the repository!
