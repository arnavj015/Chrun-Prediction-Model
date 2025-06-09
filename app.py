import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Load model and feature list
model = joblib.load('models/xgb_churn_model.pkl')
feature_list = joblib.load('models/feature_list.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title("🔮 Customer Churn Prediction Dashboard")

st.sidebar.header("📋 Customer Information")
st.sidebar.header("💰 Business Assumptions")
churn_cost = st.sidebar.number_input("Estimated loss per churned customer ($)", min_value=0, value=200)
retention_rate = st.sidebar.slider("Retention campaign success rate (%)", 0, 100, 25)



def get_user_input():
    # Basic Demographics
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior = st.sidebar.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])

    # Account Info
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly = st.sidebar.slider("Monthly Charges", 20, 120, 70)
    total = st.sidebar.slider("Total Charges", 0, 9000, 1000)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Credit card (automatic)", "Bank transfer (automatic)"
    ])
    paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

    # Services
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_sec = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_bkp = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protect = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    # Derived features
    avg_charges = total / (tenure + 1)

    raw_input = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'MultipleLines': 1 if multiple_lines == 'Yes' else 0,
        'OnlineSecurity': 1 if online_sec == 'Yes' else 0,
        'OnlineBackup': 1 if online_bkp == 'Yes' else 0,
        'DeviceProtection': 1 if device_protect == 'Yes' else 0,
        'TechSupport': 1 if tech_support == 'Yes' else 0,
        'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless == 'Yes' else 0,
        'MonthlyCharges': monthly,
        'TotalCharges': total,
        'InternetService_Fiber optic': 1 if internet == 'Fiber optic' else 0,
        'InternetService_No': 1 if internet == 'No' else 0,
        'Contract_One year': 1 if contract == 'One year' else 0,
        'Contract_Two year': 1 if contract == 'Two year' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment == 'Credit card (automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment == 'Electronic check' else 0,
        'PaymentMethod_Mailed check': 1 if payment == 'Mailed check' else 0,
        'tenure_bucket_7–12 months': 1 if 7 <= tenure <= 12 else 0,
        'tenure_bucket_13–24 months': 1 if 13 <= tenure <= 24 else 0,
        'tenure_bucket_25–48 months': 1 if 25 <= tenure <= 48 else 0,
        'tenure_bucket_49–60 months': 1 if 49 <= tenure <= 60 else 0,
        'tenure_bucket_60+ months': 1 if tenure > 60 else 0,
        'AvgCharges': avg_charges
    }

    # Create input row and align with model's expected columns
    input_df = pd.DataFrame([raw_input])
    input_df = input_df.reindex(columns=feature_list, fill_value=0)

    return input_df

# Get input
input_data = get_user_input()

# Predict churn probability
prob = model.predict_proba(input_data)[0][1]
st.subheader(f"📈 Predicted Churn Probability: {prob:.2%}")
expected_loss = prob * churn_cost
expected_saved = expected_loss * (retention_rate / 100)

# Display business metrics
st.subheader("💼 Business Impact")
col1, col2 = st.columns(2)
col1.metric("Expected Revenue Loss", f"${expected_loss:.2f}")
col2.metric("Projected Savings", f"${expected_saved:.2f}")

if prob > 0.6:
    st.warning("This customer is at high risk of churning. Consider proactive retention strategies.")
elif prob > 0.3:
    st.info("This customer has moderate churn risk. Monitor engagement and satisfaction.")
else:
    st.success("Low churn risk. Maintain satisfaction and upsell if appropriate.")



# SHAP explanation
explainer = shap.Explainer(model)
shap_values = explainer(input_data)

st.subheader("🧠 Why this prediction?")
fig, ax = plt.subplots(figsize=(10, 4))
shap.plots.waterfall(shap_values[0], max_display=10, show=False)
st.pyplot(fig)

st.markdown("---")
st.caption("Built with ❤️ using Streamlit and SHAP")
