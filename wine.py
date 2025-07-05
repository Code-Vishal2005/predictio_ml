import streamlit as st
import joblib
import numpy as np
import time
import os

# Page Configuration
st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="centered", page_icon="🍇")

# Sidebar Header
st.sidebar.markdown("## 🍷 Wine Quality Predictor")
st.sidebar.markdown("🔍 _Predict the quality of wine based on chemical properties._")
st.sidebar.info("👉 Enter values below and click **Predict** to check wine quality!")

# Centered Main Title
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color: #8B0000;'>🍷 Wine Quality Prediction App</h1>
        <h4 style='color: #555;'>Enter chemical composition of wine to check its quality</h4>
    </div>
""", unsafe_allow_html=True)

# Load the model
try:
    with open("wine_quality_model.pkl", "rb") as file:
        model = joblib.load(file)
    st.sidebar.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("❌ Model file not found!")
    st.stop()
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Sidebar Inputs
st.sidebar.subheader("🍇 Input Chemical Details")
fixed_acidity = st.sidebar.number_input("🍇 Fixed Acidity", 0.0, 20.0, step=0.1)
volatile_acidity = st.sidebar.number_input("💨 Volatile Acidity", 0.0, 2.0, step=0.01)
citric_acid = st.sidebar.number_input("🍋 Citric Acid", 0.0, 1.0, step=0.01)
residual_sugar = st.sidebar.number_input("🍬 Residual Sugar", 0.0, 20.0, step=0.1)
chlorides = st.sidebar.number_input("🧂 Chlorides", 0.0, 0.1, step=0.001)
free_sulfur_dioxide = st.sidebar.number_input("🧪 Free Sulfur Dioxide", 0, 100, step=1)
total_sulfur_dioxide = st.sidebar.number_input("🔬 Total Sulfur Dioxide", 0, 300, step=1)
density = st.sidebar.number_input("⚖️ Density", 0.990, 1.005, step=0.001)
pH = st.sidebar.number_input("🧫 pH", 2.5, 7.0, step=0.01)
sulphates = st.sidebar.number_input("🌋 Sulphates", 0.0, 2.0, step=0.01)
alcohol = st.sidebar.number_input("🍷 Alcohol %", 0.0, 20.0, step=0.1)

# Predict Button
if st.button("🔍 Predict Wine Quality"):
    with st.spinner("Analyzing wine quality... 🍷"):
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

    # Prepare and predict
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                            chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                            density, pH, sulphates, alcohol]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    # Display Prediction
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🍷 Wine Quality Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🧾 Prediction Result</h3>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.success("🍷 This wine is likely **GOOD QUALITY**! 🎉")
        st.markdown(f"<div style='text-align: center; font-size: 20px;'>✅ Confidence: <strong>{int(prediction_proba[1]*100)}%</strong> 🎯</div>", unsafe_allow_html=True)
        st.snow()
    else:
        st.error("⚠️ This wine might be **LOW QUALITY**.")
        st.markdown(f"<div style='text-align: center; font-size: 20px;'>💔 Confidence: <strong>{int(prediction_proba[0]*100)}%</strong></div>", unsafe_allow_html=True)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About the App")
st.sidebar.info("""
- 🧠 Built using a Machine Learning model (SVC)
- 📊 Trained on UCI Wine Quality Dataset
- 💡 Helps winemakers assess wine quality
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by **Vishal Kumar**")
st.sidebar.markdown("📧 [LinkedIn](https://www.linkedin.com/in/vishalkumar2003/)")

# Footer Note
st.markdown("---")
st.info("🔁 Try changing values in the sidebar to test different wine samples!")
