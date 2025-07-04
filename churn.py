import streamlit as st
import joblib
import pandas as pd

# Load the model
loaded = joblib.load("E:\\Machine Learning Projects\\churn_model.pkl")
model = loaded['model'] if isinstance(loaded, dict) and 'model' in loaded else loaded

# Page configuration
st.set_page_config(page_title="📉 Churn Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📉 Customer Churn Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>💼 Use the sidebar to input customer details and predict the likelihood of churn</p>", unsafe_allow_html=True)

# Sidebar input
st.sidebar.markdown("## 🧾 Customer Info")
gender = st.sidebar.selectbox("👤 Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("👴 Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("❤️ Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("👶 Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("📆 Tenure (months)", min_value=0, max_value=100, value=1)
PhoneService = st.sidebar.selectbox("📱 Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("📶 Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.sidebar.selectbox("🌐 Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.sidebar.selectbox("🔐 Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.sidebar.selectbox("💾 Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.sidebar.selectbox("🛡️ Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.sidebar.selectbox("👨‍💻 Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.sidebar.selectbox("📺 Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.sidebar.selectbox("🎬 Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.sidebar.selectbox("📄 Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("📧 Paperless Billing", ["Yes", "No"])
PaymentMethod = st.sidebar.selectbox("💳 Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
MonthlyCharges = st.sidebar.number_input("💸 Monthly Charges", min_value=0.0)
TotalCharges = st.sidebar.number_input("💰 Total Charges", min_value=0.0)

# Mapping for encoding
mapping = {
    'gender': {'Male': 0, 'Female': 1},
    'Partner': {'Yes': 1, 'No': 0},
    'Dependents': {'Yes': 1, 'No': 0},
    'PhoneService': {'Yes': 1, 'No': 0},
    'MultipleLines': {'No phone service': 0, 'No': 1, 'Yes': 2},
    'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
    'OnlineSecurity': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'OnlineBackup': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'DeviceProtection': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'TechSupport': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingTV': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'StreamingMovies': {'No': 0, 'Yes': 1, 'No internet service': 2},
    'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
    'PaperlessBilling': {'Yes': 1, 'No': 0},
    'PaymentMethod': {
        'Electronic check': 0,
        'Mailed check': 1,
        'Bank transfer (automatic)': 2,
        'Credit card (automatic)': 3
    }
}

# Predict button
if st.button("🔮 Predict Now!"):
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    # Apply encoding
    for key in mapping:
        input_data[key] = mapping[key][input_data[key]]

    input_df = pd.DataFrame([input_data])

    try:
        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1] * 100

        st.markdown("## 🧠 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ The customer is likely to **CHURN**.")
        else:
            st.success(f"✅ The customer is likely to **STAY**.")
        st.info(f"📈 Churn Probability: **{prob:.2f}%**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
