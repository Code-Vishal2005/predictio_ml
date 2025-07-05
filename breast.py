import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="Breast Cancer Disease Prediction", layout="centered", page_icon="🎀💗")

# Sidebar header
st.sidebar.title(" 🎀 Breast Cancer Disease Predictor")
st.sidebar.markdown("🔍 This app predicts the likelihood of breast cancer disease based on patient input.")
st.sidebar.info("👉 Fill in the details on the main screen and click **Predict**.")

# Main Title
st.markdown("<h1 style='text-align: center; color: pink;'>🎀🎗️ Breast Cancer Disease Prediction App</h1>", unsafe_allow_html=True)

# Load the model
try:
    with open("Breast_model.pkl", "rb") as vishal:
        model = joblib.load(vishal)
    st.sidebar.success("✅ Model loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("❌ Model file not found!")
    st.stop()
except Exception as e:
    st.sidebar.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Input columns
st.sidebar.subheader("🩺 Input Patient Details")
col1, col2, col3 = st.sidebar.columns(3)

with col1:
    mean_radius = col1.number_input("Mean Radius", 0.0, 30.0, step=0.1)
    mean_texture = col1.number_input("Mean Texture", 0.0, 40.0, step=0.1)
    mean_perimeter = col1.number_input("Mean Perimeter", 0.0, 200.0, step=0.1)
    mean_area = col1.number_input("Mean Area", 0.0, 5000.0, step=0.1)
    mean_smoothness = col1.number_input("Mean Smoothness", 0.0, 1.0, step=0.01)
    mean_compactness = col1.number_input("Mean Compactness", 0.0, 1.0, step=0.01)
    mean_concavity = col1.number_input("Mean Concavity", 0.0, 1.0, step=0.01)
    mean_concave_points = col1.number_input("Mean Concave Points", 0.0, 1.0, step=0.01)
    mean_symmetry = col1.number_input("Mean Symmetry", 0.0, 1.0, step=0.01)
    mean_fractal_dimension = col1.number_input("Mean Fractal Dimension", 0.0, 1.0, step=0.01)

with col2:
    radius_error = col2.number_input("Radius Error", 0.0, 10.0, step=0.1)
    texture_error = col2.number_input("Texture Error", 0.0, 10.0, step=0.1)
    perimeter_error = col2.number_input("Perimeter Error", 0.0, 10.0, step=0.1)
    area_error = col2.number_input("Area Error", 0.0, 1000.0, step=0.1)
    smoothness_error = col2.number_input("Smoothness Error", 0.0, 0.1, step=0.001)
    compactness_error = col2.number_input("Compactness Error", 0.0, 0.1, step=0.001)
    concavity_error = col2.number_input("Concavity Error", 0.0, 0.1, step=0.001)
    concave_points_error = col2.number_input("Concave Points Error", 0.0, 0.1, step=0.001)
    symmetry_error = col2.number_input("Symmetry Error", 0.0, 0.1, step=0.001)
    fractal_dimension_error = col2.number_input("Fractal Dimension Error", 0.0, 0.1, step=0.001)

with col3:
    worst_radius = col3.number_input("Worst Radius", 0.0, 30.0, step=0.1)
    worst_texture = col3.number_input("Worst Texture", 0.0, 40.0, step=0.1)
    worst_perimeter = col3.number_input("Worst Perimeter", 0.0, 200.0, step=0.1)
    worst_area = col3.number_input("Worst Area", 0.0, 5000.0, step=0.1)
    worst_smoothness = col3.number_input("Worst Smoothness", 0.0, 1.0, step=0.01)
    worst_compactness = col3.number_input("Worst Compactness", 0.0, 1.0, step=0.01)
    worst_concavity = col3.number_input("Worst Concavity", 0.0, 1.0, step=0.01)
    worst_concave_points = col3.number_input("Worst Concave Points", 0.0, 1.0, step=0.01)
    worst_symmetry = col3.number_input("Worst Symmetry", 0.0, 1.0, step=0.01)
    worst_fractal_dimension = col3.number_input("Worst Fractal Dimension", 0.0, 1.0, step=0.01)

# Predict button
if st.button("🔍 Predict Breast Cancer Disease"):
    input_data = np.array([[mean_radius, mean_texture, mean_perimeter, mean_area,
                            mean_smoothness, mean_compactness, mean_concavity,
                            mean_concave_points, mean_symmetry, mean_fractal_dimension,
                            radius_error, texture_error, perimeter_error, area_error,
                            smoothness_error, compactness_error, concavity_error,
                            concave_points_error, symmetry_error, fractal_dimension_error,
                            worst_radius, worst_texture, worst_perimeter, worst_area,
                            worst_smoothness, worst_compactness, worst_concavity,
                            worst_concave_points, worst_symmetry, worst_fractal_dimension]])

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0]

    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>🎀 Breast Cancer Disease Prediction</h2>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🧾 Prediction Result</h3>", unsafe_allow_html=True)

    if prediction[0] == 0:
        st.success("✅ The model predicts Benign (Healthy) , Enjoy Your Life!")
        st.balloons()
    else:
        st.error("⚠️ The model predicts Malignant (Cancerous) , Please Consult The Doctor.")
        st.snow()

    # Show prediction probability
    st.markdown(f"""
    <h4 style='text-align: center;'>
    🎯 <u>Prediction Probability</u><br>
    <span style='color: pink;'>🧬 Cancer (Malignant),Please Consult The Doctor): {prediction_proba[1]*100:.2f}%</span> |
    <span style='color: green;'> ✅ Healthy Breast (Benign) , Enjoy Your Life: {prediction_proba[0]*100:.2f}%</span>
    </h4>
    """, unsafe_allow_html=True)

# Sidebar footer
st.sidebar.markdown("---") 
st.sidebar.markdown("### ℹ️ About the App")
st.sidebar.info(
    """
    🎀 **About This App**  
    This app predicts whether a breast tumor is **Benign (Healthy)** or **Malignant (Cancerous)** using 30 medical input features.

    🧠 Powered by a machine learning model trained on real-world data.

    📌 Tip: Use accurate values for better prediction results.
    """
)
