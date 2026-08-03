import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Title and description
st.title("🩺 Diabetes Prediction App")
st.write(
    "Provide the required physiological metrics below to predict the likelihood of diabetes."
)

# Load the trained model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.subheader("Input Features")

# Create two columns for user inputs
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1, step=1)
    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0.0, max_value=900.0, value=79.0, step=1.0)
    bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
    age = st.number_input("Age", min_value=1, max_value=120, value=33, step=1)

st.markdown("---")

# Prediction button
if st.button("Predict Diabetes Status", type="primary", use_container_width=True):
    # Prepare input data matching feature names expected by the model
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age
    }])

    # Make prediction and obtain probability if supported
    prediction = model.predict(input_data)[0]
    
    st.subheader("Results")
    if prediction == 1:
        st.error("⚠️ **High Risk:** The model predicts that the patient is likely to have diabetes.")
    else:
        st.success("✅ **Low Risk:** The model predicts that the patient is unlikely to have diabetes.")
        
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        st.write(f"**Confidence Scores:**")
        st.write(f"- Non-Diabetic: `{probabilities[0] * 100:.2f}%`")
        st.write(f"- Diabetic: `{probabilities[1] * 100:.2f}%`")
