# Telco Customer Churn Prediction

## 📌 Project Overview
This project aims to predict customer churn in a telecommunications company using machine learning. Churn prediction helps identify customers likely to stop using the service, enabling proactive retention strategies.

We use the Telco Customer Churn dataset to explore customer demographics, service usage, and billing details, then build a classification model to predict churn likelihood.

---

## 📂 Dataset
**File:** `Telco_customer_churn.xlsx`  

**Source:** Public dataset (commonly used for churn analysis)  

**Description:** Contains customer-level data including:
- Customer demographics (gender, senior citizen, partner, dependents)
- Service details (phone, internet, streaming, security services)
- Account information (contract type, tenure, billing method)  
- **Target variable:** Churn — Yes/No

---

## ⚙️ Tech Stack
**Language:** Python 3.x  

**Libraries:**
- `pandas` — Data manipulation
- `numpy` — Numerical operations
- `matplotlib` / `seaborn` — Visualization
- `scikit-learn` — Machine learning models
- `openpyxl` — Excel file handling

---

## 🚀 Project Workflow
1. **Data Ingestion** — Load Excel data into Pandas.  
2. **Exploratory Data Analysis (EDA)** — Understand patterns, correlations, and missing values.  
3. **Data Preprocessing** — Encode categorical variables, scale numerical features, handle missing data.  
4. **Model Training** — Train and evaluate classification models (Logistic Regression, Random Forest, etc.).  
5. **Model Evaluation** — Use accuracy, precision, recall, F1-score, and ROC-AUC.  
6. **Deployment (Optional)** — Deploy using Streamlit or Flask.

---

## 📊 Example Use Case
The churn model can help a telecom company:
- Identify high-risk customers.
- Offer discounts or incentives.
- Improve retention rate and revenue.

---

## 📜 License
This project uses open-source datasets. You are free to use and modify the code for educational purposes.
