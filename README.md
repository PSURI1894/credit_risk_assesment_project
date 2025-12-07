# 💳 AI-Based Credit Risk Assessment System

### 🔍 Predicting Loan Default Probability using Financial & Credit History Data

## 📘 Overview

The **Credit Risk Assessment System** is a full end-to-end Data Science project that predicts whether a customer is likely to **default on a loan** based on demographic, employment, income, and credit history data.

The goal is to assist banks and financial institutions in automating credit approval and minimizing risk exposure through **data-driven decision making**.

---

## 🚀 Live Demo

🔗 **Streamlit App:** [Deployed on Streamlit Cloud](https://share.streamlit.io/) *(add your deployed app link here)*  
🧠 **Model:** Random Forest trained on preprocessed credit application and repayment data.

---

## 🧩 Key Features

✅ Predict loan default probability for new or existing customers  
✅ Interactive Streamlit dashboard with customer exploration  
✅ Real-time **"What-if" simulator** to adjust applicant parameters  
✅ Model explainability through **feature importance** and **SHAP** analysis  
✅ Handles **class imbalance** using SMOTE oversampling  
✅ Clean, well-documented Jupyter notebook for full reproducibility  

---

## 📂 Repository Structure

```
credit_risk_assesment_project/
│
├── app.py                             # Streamlit app for deployment
├── credit_risk_assessment_project.ipynb # Full modeling + EDA notebook
├── application_record.csv              # Raw application dataset
├── credit_record.csv                   # Raw credit history dataset
├── clean_credit_risk_dataset.csv       # Final merged + engineered dataset
│
├── credit_risk_rf_model.pkl            # Trained Random Forest model
├── credit_risk_scaler.pkl              # Feature scaler (for LR compatibility)
├── credit_risk_label_encoders.pkl      # Label encoders for categorical vars
│
├── requirements.txt                    # Python dependencies
└── README.md                           # You're reading it 🙂
```

---

## 🧠 Data Description

The project uses two datasets:

| Dataset | Description |
|----------|--------------|
| `application_record.csv` | Customer demographics and employment info (age, income, marital status, etc.) |
| `credit_record.csv` | Monthly credit repayment statuses with `STATUS` codes |

**STATUS Code Meaning:**

| Code | Meaning |
|------|----------|
| 0 | Paid on time |
| 1–5 | Payment late by 1–5 months |
| C | Contract closed |
| X | No loan for that month |

A customer is labeled as a **defaulter (`default_flag = 1`)** if they have ever had `STATUS ≥ 2`.

---

## 🧮 Methodology

### 1️⃣ Data Preprocessing & Feature Engineering
- Cleaned missing and inconsistent records  
- Converted day-based columns into interpretable features:  
  - `DAYS_BIRTH → AGE_YEARS`  
  - `DAYS_EMPLOYED → YEARS_EMPLOYED`  
- Engineered financial ratios:
  - `INCOME_PER_CHILD`
  - `INCOME_PER_FAM_MEMBER`
- Aggregated credit behavior from `credit_record`:  
  - `num_delay_months`, `delay_ratio`, `max_delay_severity`, `credit_history_length`

### 2️⃣ Label Construction
A `default_flag` target variable was created using credit repayment behavior:

```python
if any(STATUS in ['2','3','4','5']):
    default_flag = 1
else:
    default_flag = 0
```

### 3️⃣ Handling Class Imbalance
Used **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.

### 4️⃣ Model Training
Two models were trained and compared:

| Model | ROC-AUC | PR-AUC | Notes |
|--------|----------|---------|-------|
| Logistic Regression | ~0.82 | ~0.65 | Fast baseline |
| Random Forest | **~0.91** | **~0.73** | Best performing model |

### 5️⃣ Model Evaluation
Metrics:
- ROC–AUC  
- Precision–Recall AUC  
- F1-Score  
- Confusion Matrix  

> Business-oriented threshold tuning prioritized **recall on defaulters** (catch risky cases).

### 6️⃣ Model Explainability
- **Feature Importance** (global)  
- **SHAP (SHapley Additive exPlanations)** for transparency into model predictions.

---

## 🧭 Streamlit App Functionality

### **1️⃣ Explore Existing Customers**
- Choose any customer from dataset  
- View full profile, model-predicted default probability, and actual outcome

### **2️⃣ What-if Simulation**
- Modify applicant parameters interactively (Age, Income, Dependents, Credit Behavior)
- Instantly see new predicted risk probability

### **3️⃣ Model Info**
- Dataset summary, target distribution, feature importance plots, and methodology overview

---

## ⚙️ Installation & Local Setup

### Clone Repository
```bash
git clone https://github.com/PSURI1894/credit_risk_assesment_project.git
cd credit_risk_assesment_project
```

### Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
source venv/bin/activate # Mac/Linux
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Streamlit App
```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🌐 Deployment (Streamlit Cloud)

Deployed easily via **[Streamlit Community Cloud](https://share.streamlit.io/)**

1. Push this repo to GitHub  
2. Go to Streamlit Cloud → “New app”  
3. Repo = `PSURI1894/credit_risk_assesment_project`  
4. Branch = `main`  
5. File = `app.py`  
6. Hit **Deploy**

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.10 |
| **Libraries** | pandas, numpy, scikit-learn, shap, imbalanced-learn, seaborn |
| **Visualization** | matplotlib, seaborn, Plotly |
| **Machine Learning** | Random Forest, Logistic Regression |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git + GitHub |

---

## 💡 Future Enhancements

- Integrate **XGBoost / LightGBM**  
- Add **Fairness Metrics** (bias detection)  
- Deploy via **FastAPI or Flask API**  
- Connect to **SQL / NoSQL databases**  
- Build automated **credit decision recommendation system**

---

## 👤 Author

**Parth Suri**  
🎓 Data Science Enthusiast | Machine Learning | Cloud Deployment  
📍 VIT Vellore  
🔗 [GitHub](https://github.com/PSURI1894) • [LinkedIn](https://linkedin.com/in/)  

---

## 🏁 License

Licensed under the **MIT License** — free to use and modify with attribution.

---

### ⭐ If you found this helpful, give the repo a star!
