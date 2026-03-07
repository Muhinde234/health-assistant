
import streamlit as st
import joblib
import numpy as np
import os

# Professional styling
st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

model_path = os.path.join(os.path.dirname(__file__), "../../models/heart_model.pkl")
model = joblib.load(model_path)

# Custom CSS for professional styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 2rem;
    }
    
    /* Title styling */
    h1 {
        color: #00A8E8;
        text-align: center;
        margin-bottom: 1rem;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Input sections */
    .input-section {
        background-color: #E8F4F8;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #00A8E8;
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #00A8E8 !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        padding: 0.75rem !important;
        border-radius: 8px !important;
        border: none !important;
    }
    
    .stButton>button:hover {
        background-color: #0091C9 !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ AI Heart Disease Risk Assessment")

st.markdown(
    """
    <div class="subtitle">
    Advanced ML-based system for predicting heart disease risk<br>
    <small>Powered by medical data analysis and machine learning</small>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Patient Information")
    age = st.slider("Age (years)", 20, 100, 50)
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
    cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
    chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=400, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    st.markdown("### Medical Measurements")
    restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
    thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
    exang = st.selectbox("Exercise Induced Angina", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1)
    slope = st.selectbox("ST Segment Slope", options=[0, 1, 2], format_func=lambda x: f"Type {x}")
    ca = st.selectbox("Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], format_func=lambda x: f"Type {x}")

st.divider()

# Prediction button
col_button = st.columns([1, 2, 1])
with col_button[1]:
    if st.button("🔍 Predict Risk", use_container_width=True):
        # Create feature array in correct order
        patient = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        prediction = model.predict(patient)
        risk_prob = model.predict_proba(patient)
        
        risk_percentage = risk_prob[0][1] * 100

        st.divider()
        
        # Display results with professional styling
        if prediction[0] == 1:
            st.error(f"### ⚠️ HIGH RISK DETECTED")
            st.markdown(f"""
            <div style='background-color: #FFE5E5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #E74C3C;'>
                <h4 style='color: #C0392B; margin: 0;'>Risk Level: {risk_percentage:.1f}%</h4>
                <p style='color: #555; margin: 0.5rem 0 0 0;'><strong>Recommendation:</strong> Please consult with a healthcare professional immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"### ✅ LOW RISK")
            st.markdown(f"""
            <div style='background-color: #E5F5E5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #27AE60;'>
                <h4 style='color: #1E8449; margin: 0;'>Risk Level: {risk_percentage:.1f}%</h4>
                <p style='color: #555; margin: 0.5rem 0 0 0;'><strong>Recommendation:</strong> Continue with regular health check-ups and maintain a healthy lifestyle.</p>
            </div>
            """, unsafe_allow_html=True)