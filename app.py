import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Analyzer",
    page_icon="🩺",
    layout="centered"
)

# Load the trained KNN model
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

# Custom CSS for Background Image and Styled UI Cards
bg_image_url = "https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1920&q=80"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.65), rgba(0, 0, 0, 0.65)), url("{bg_image_url}");
        background-attachment: fixed;
        background-size: cover;
    }}
    
    /* Content Container styling */
    .block-container {{
        background: rgba(255, 255, 255, 0.92);
        padding: 2.5rem;
        border-radius: 16px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }}

    /* Title styling */
    h1 {{
        color: #1A365D;
        text-align: center;
        font-weight: 700;
    }}

    /* Button styling */
    .stButton>button {{
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.6rem 1rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🩺 Diabetes Risk Predictor")
st.markdown("<p style='text-align: center; color: #4A5568;'>Select the physiological categories below to analyze diabetes risk.</p>", unsafe_allow_html=True)
st.markdown("---")

st.subheader("📋 Patient Profile")

col1, col2 = st.columns(2)

with col1:
    # Pregnancies (Categorical)
    preg_cat = st.selectbox(
        "Pregnancies",
        options=["None (0)", "1 to 2", "3 to 5", "More than 5"]
    )
    
    # Glucose Level (Categorical)
    glucose_cat = st.selectbox(
        "Fasting Glucose Level",
        options=[
            "Normal (< 100 mg/dL)",
            "Prediabetes (100 - 125 mg/dL)",
            "High / Diabetic (126 - 160 mg/dL)",
            "Very High (> 160 mg/dL)"
        ]
    )

    # Blood Pressure (Categorical)
    bp_cat = st.selectbox(
        "Blood Pressure Status",
        options=[
            "Normal (Systolic/Diastolic Standard ~ 70 mmHg)",
            "Elevated (~ 80 mmHg)",
            "Stage 1 Hypertension (~ 90 mmHg)",
            "Stage 2 / High (~ 100+ mmHg)"
        ]
    )

    # Skin Thickness (Categorical)
    skin_cat = st.selectbox(
        "Triceps Skin Fold Thickness",
        options=["Low / Lean (< 15 mm)", "Average (15 - 30 mm)", "High (> 30 mm)"]
    )

with col2:
    # Insulin Level (Categorical)
    insulin_cat = st.selectbox(
        "Insulin Level Status",
        options=[
            "Normal / Low (~ 30 - 80 mu U/ml)",
            "Moderate (~ 81 - 150 mu U/ml)",
            "High (~ 151 - 250 mu U/ml)",
            "Very High (> 250 mu U/ml)"
        ]
    )

    # BMI Category
    bmi_cat = st.selectbox(
        "Body Mass Index (BMI)",
        options=[
            "Underweight (< 18.5)",
            "Normal weight (18.5 - 24.9)",
            "Overweight (25 - 29.9)",
            "Obese Class I (30 - 34.9)",
            "Obese Class II / III (≥ 35)"
        ]
    )

    # Diabetes Pedigree Function (Family History)
    dpf_cat = st.selectbox(
        "Family History of Diabetes",
        options=[
            "No Known Family History (Low Risk)",
            "Moderate Family History (Moderate Risk)",
            "Strong Family History (High Risk)"
        ]
    )

    # Age Groups
    age_cat = st.selectbox(
        "Age Group",
        options=["Young Adult (18 - 29)", "Adult (30 - 44)", "Middle Aged (45 - 59)", "Senior (60+)"]
    )

# Mapping Categorical selections to approximate numerical values expected by model
preg_map = {"None (0)": 0, "1 to 2": 1, "3 to 5": 4, "More than 5": 7}
glucose_map = {
    "Normal (< 100 mg/dL)": 85.0,
    "Prediabetes (100 - 125 mg/dL)": 112.0,
    "High / Diabetic (126 - 160 mg/dL)": 143.0,
    "Very High (> 160 mg/dL)": 175.0
}
bp_map = {
    "Normal (Systolic/Diastolic Standard ~ 70 mmHg)": 70.0,
    "Elevated (~ 80 mmHg)": 80.0,
    "Stage 1 Hypertension (~ 90 mmHg)": 90.0,
    "Stage 2 / High (~ 100+ mmHg)": 105.0
}
skin_map = {"Low / Lean (< 15 mm)": 10.0, "Average (15 - 30 mm)": 23.0, "High (> 30 mm)": 35.0}
insulin_map = {
    "Normal / Low (~ 30 - 80 mu U/ml)": 55.0,
    "Moderate (~ 81 - 150 mu U/ml)": 115.0,
    "High (~ 151 - 250 mu U/ml)": 200.0,
    "Very High (> 250 mu U/ml)": 320.0
}
bmi_map = {
    "Underweight (< 18.5)": 17.5,
    "Normal weight (18.5 - 24.9)": 22.0,
    "Overweight (25 - 29.9)": 27.5,
    "Obese Class I (30 - 34.9)": 32.5,
    "Obese Class II / III (≥ 35)": 38.0
}
dpf_map = {
    "No Known Family History (Low Risk)": 0.2,
    "Moderate Family History (Moderate Risk)": 0.5,
    "Strong Family History (High Risk)": 0.9
}
age_map = {
    "Young Adult (18 - 29)": 24,
    "Adult (30 - 44)": 37,
    "Middle Aged (45 - 59)": 52,
    "Senior (60+)": 65
}

st.markdown("---")

# Prediction action
if st.button("🔍 Analyze Risk Profile", type="primary", use_container_width=True):
    # Construct input dataframe matching exact model feature structure
    input_data = pd.DataFrame([{
        "Pregnancies": preg_map[preg_cat],
        "Glucose": glucose_map[glucose_cat],
        "BloodPressure": bp_map[bp_cat],
        "SkinThickness": skin_map[skin_cat],
        "Insulin": insulin_map[insulin_cat],
        "BMI": bmi_map[bmi_cat],
        "DiabetesPedigreeFunction": dpf_map[dpf_cat],
        "Age": age_map[age_cat]
    }])

    prediction = model.predict(input_data)[0]
    
    st.subheader("Results")
    if prediction == 1:
        st.error("⚠️ **High Risk Detected:** The metrics indicate a high likelihood of diabetes.")
    else:
        st.success("✅ **Low Risk Detected:** The metrics indicate a low likelihood of diabetes.")
        
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        st.write("### Confidence Breakdown")
        st.progress(float(probabilities[1]))
        st.write(f"- **Low Risk Confidence:** `{probabilities[0] * 100:.1f}%`")
        st.write(f"- **High Risk Confidence:** `{probabilities[1] * 100:.1f}%`")
