# Customer Churn Prediction Dashboard

A full-stack machine learning project that predicts telecom customer churn with actionable insights. Built with XGBoost, SHAP for model explainability, and Streamlit to simulate churn risk and business impact.

---

## What It Does

- Predicts if a customer is likely to churn
- Explains *why* using SHAP interpretability
- Estimates business cost of churn
- Simulates how much you can save by acting early

---

## Key Features

- Cleaned & engineered real-world telecom dataset
- Tuned XGBoost classifier with `scale_pos_weight` for imbalance
- SHAP-based feature insights (global + per-customer)
- Streamlit dashboard with churn simulation + expected loss
- Modularized for easy extension or API deployment

---

## 📂 Folder Structure

customer-churn-prediction/
│
├── app/ # Streamlit dashboard
│ └── streamlit_app.py
├── model/ # Saved XGBoost model + feature list
│ ├── xgb_model.pkl
│ └── feature_list.pkl
├── notebooks/ # EDA and model building
│ └── churn_model_building.ipynb
├── data/ # Optional sample data
├── README.md
└── requirements.txt



## Insights Learned

- Tenure and Monthly Charges are key drivers of churn
- Fiber optic users on monthly contracts churn significantly more
- Electronic check users have higher churn risk than other payment methods

---

## 🧾 How to Run It

1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/customer-churn-prediction.git
   cd customer-churn-prediction

Launch dashboard:

streamlit run app/streamlit_app.py

Built With
Python, Pandas, Scikit-learn, XGBoost

SHAP for model explainability

Streamlit for dashboard UI

👋 About Me
Hi, I’m Arnav Jain — a Data Science & Economics graduate from UC Berkeley. I enjoy building data tools that combine ML with real business context.

