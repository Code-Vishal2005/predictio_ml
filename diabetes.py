import streamlit as st
import joblib
import numpy as np
from sklearn.svm import SVC
import time

# Load the trained model
with open("trained_model.pkl", "rb") as file:
    model = joblib.load(file)

# Page Configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🧬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f9ff;
        }
        h1 {
            color: #006d77;
        }
        .stButton>button {
            background-color: #006d77;
            color: white;
            font-weight: bold;
            border-radius: 8px;
        }
        .stButton>button:hover {
            background-color: #83c5be;
            color: black;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center;'>🧬 Diabetes Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Get your diabetes risk percentage with ML 🚑</p>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📝 Patient Information")
    st.info("Please enter the patient's health details below:")

    pregnancies = st.number_input("🤰 Pregnancies", min_value=0, step=1)
    glucose = st.number_input("🩸 Glucose Level", min_value=0)
    blood_pressure = st.number_input("💉 Blood Pressure", min_value=0)
    skin_thickness = st.number_input("📏 Skin Thickness", min_value=0)
    insulin = st.number_input("🧪 Insulin Level", min_value=0)
    bmi = st.number_input("⚖️ BMI (Body Mass Index)", min_value=0.0, format="%.2f")
    dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0, format="%.3f")
    age = st.number_input("🎂 Age", min_value=0, step=1)

    st.markdown("---")
    predict_button = st.button("🔍 Predict")

# Prediction
if predict_button:
    with st.spinner("🔍 Analyzing health parameters..."):
        time.sleep(2)  # Simulate processing time

    st.subheader("📊 Prediction Result")

    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.markdown(f"### 📈 **Prediction: Diabetes**")
        st.error(f"⚠️ High Risk of Diabetes Detected!")
        st.markdown(f"### 🔴 **Probability: {probability:.2f}%**")
        st.markdown("💡 Please consult a healthcare professional.")
        st.image("https://media.giphy.com/media/fAnEC88LccN7a/giphy.gif", caption="Warning Signal", use_column_width=True)

    else:
        st.markdown(f"### 📈 **Prediction: No Diabetes**")
        st.success(f"✅ No Risk of Diabetes Detected.")
        st.markdown(f"### 🟢 **Probability: {probability:.2f}%**")
        st.markdown("🎉 Keep up your healthy lifestyle!")
        
        # Flower/Confetti Animation
        st.balloons()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 14px;'>🧠 Developed with ❤️ by <b>Vishal Kumar</b><br>📚 Powered by Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
