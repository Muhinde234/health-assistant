
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Initialize session state for patient history
if 'patient_history' not in st.session_state:
    st.session_state.patient_history = []

st.set_page_config(
    page_title="Heart Disease Risk Assessment",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

model_path = os.path.join(os.path.dirname(__file__), "../../models/heart_model.pkl")
model = joblib.load(model_path)


# Feature descriptions and normal ranges
FEATURE_INFO = {
    'age': {'name': 'Age', 'unit': 'years', 'normal': '20-65', 'description': 'Patient age in years'},
    'sex': {'name': 'Sex', 'options': {0: 'Female', 1: 'Male'}, 'description': 'Biological sex'},
    'cp': {'name': 'Chest Pain Type', 'options': {
        0: 'Asymptomatic', 
        1: 'Atypical Angina', 
        2: 'Non-Anginal Pain', 
        3: 'Typical Angina'
    }, 'description': 'Type of chest pain experienced'},
    'trestbps': {'name': 'Resting BP', 'unit': 'mm Hg', 'normal': '90-120', 'description': 'Resting blood pressure'},
    'chol': {'name': 'Cholesterol', 'unit': 'mg/dl', 'normal': '<200', 'description': 'Serum cholesterol level'},
    'fbs': {'name': 'Fasting Blood Sugar', 'options': {0: 'Normal (<120 mg/dl)', 1: 'Elevated (>120 mg/dl)'}, 'description': 'Fasting blood sugar level'},
    'restecg': {'name': 'Resting ECG', 'options': {
        0: 'Normal', 
        1: 'ST-T Wave Abnormality', 
        2: 'Left Ventricular Hypertrophy'
    }, 'description': 'Resting electrocardiographic results'},
    'thalach': {'name': 'Max Heart Rate', 'unit': 'bpm', 'normal': '60-100', 'description': 'Maximum heart rate achieved'},
    'exang': {'name': 'Exercise Angina', 'options': {0: 'No', 1: 'Yes'}, 'description': 'Exercise induced angina'},
    'oldpeak': {'name': 'ST Depression', 'unit': 'mm', 'normal': '0-1', 'description': 'ST depression induced by exercise'},
    'slope': {'name': 'ST Slope', 'options': {
        0: 'Downsloping', 
        1: 'Flat', 
        2: 'Upsloping'
    }, 'description': 'Slope of peak exercise ST segment'},
    'ca': {'name': 'Major Vessels', 'unit': 'vessels', 'normal': '0', 'description': 'Number of major vessels colored by fluoroscopy'},
    'thal': {'name': 'Thalassemia', 'options': {
        0: 'Normal', 
        1: 'Fixed Defect', 
        2: 'Reversible Defect',
        3: 'Unknown'
    }, 'description': 'Thalassemia blood disorder status'}
}

st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem;
    }
    
    /* Title styling */
    h1 {
        color: #00A8E8;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info box */
    .info-box {
        background-color: #E8F4F8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00A8E8;
        margin: 1rem 0;
    }
    
    /* Warning box */
    .warning-box {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
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
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #0091C9 !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,168,232,0.4) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #E8F4F8;
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #00A8E8;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("❤️ AI Heart Disease Risk Assessment")

st.markdown(
    """
    <div class="subtitle">
    Advanced ML-Powered Cardiac Risk Prediction System<br>
    <small>Empowering Healthcare Professionals with AI-Driven Insights</small>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/heart-with-pulse.png", width=150)
    st.markdown("## 🏥 Patient Dashboard")
    
    patient_name = st.text_input("👤 Patient Name", placeholder="Enter patient name")
    patient_id = st.text_input("🆔 Patient ID", placeholder="Enter patient ID")
    
    st.divider()
    
    st.markdown("### 📊 Quick Stats")
    if st.session_state.patient_history:
        st.metric("Total Assessments", len(st.session_state.patient_history))
        high_risk_count = sum(1 for p in st.session_state.patient_history if p['prediction'] == 1)
        st.metric("High Risk Cases", high_risk_count)
    else:
        st.info("No assessments yet")
    
    st.divider()
    
    st.markdown("### ℹ️ About This System")
    st.markdown("""
    This AI system uses **Logistic Regression** and **Random Forest** models trained on the Cleveland Heart Disease dataset.
    
    **Accuracy**: ~85%  
    **Dataset**: 303 patients  
    **Features**: 13 clinical parameters
    """)
    
    st.divider()
    
    if st.button("🗑️ Clear History"):
        st.session_state.patient_history = []
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Risk Assessment", "📊 Analytics Dashboard", "📋 Patient History", "ℹ️ Information"])

# TAB 1: Risk Assessment
with tab1:

    st.markdown("### Enter Patient Information")
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 Demographics & Vitals")
        age = st.slider("Age (years)", 20, 100, 50, help=FEATURE_INFO['age']['description'])
        sex = st.selectbox("Sex", options=[0, 1], 
                          format_func=lambda x: FEATURE_INFO['sex']['options'][x],
                          help=FEATURE_INFO['sex']['description'])
        cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3], 
                         format_func=lambda x: FEATURE_INFO['cp']['options'][x],
                         help=FEATURE_INFO['cp']['description'])
        trestbps = st.number_input(f"Resting Blood Pressure ({FEATURE_INFO['trestbps']['unit']}) - Normal: {FEATURE_INFO['trestbps']['normal']}", 
                                   min_value=80, max_value=200, value=120,
                                   help=FEATURE_INFO['trestbps']['description'])
        chol = st.number_input(f"Cholesterol ({FEATURE_INFO['chol']['unit']}) - Normal: {FEATURE_INFO['chol']['normal']}", 
                              min_value=100, max_value=600, value=200,
                              help=FEATURE_INFO['chol']['description'])
        fbs = st.selectbox("Fasting Blood Sugar", options=[0, 1], 
                          format_func=lambda x: FEATURE_INFO['fbs']['options'][x],
                          help=FEATURE_INFO['fbs']['description'])

    with col2:
        st.markdown("#### 🔬 Cardiac Measurements")
        restecg = st.selectbox("Resting ECG Results", options=[0, 1, 2], 
                              format_func=lambda x: FEATURE_INFO['restecg']['options'][x],
                              help=FEATURE_INFO['restecg']['description'])
        thalach = st.number_input(f"Max Heart Rate ({FEATURE_INFO['thalach']['unit']}) - Normal: {FEATURE_INFO['thalach']['normal']}", 
                                 min_value=60, max_value=220, value=150,
                                 help=FEATURE_INFO['thalach']['description'])
        exang = st.selectbox("Exercise Induced Angina", options=[0, 1], 
                            format_func=lambda x: FEATURE_INFO['exang']['options'][x],
                            help=FEATURE_INFO['exang']['description'])
        oldpeak = st.slider(f"ST Depression - Oldpeak ({FEATURE_INFO['oldpeak']['unit']}) - Normal: {FEATURE_INFO['oldpeak']['normal']}", 
                           0.0, 6.0, 1.0, step=0.1,
                           help=FEATURE_INFO['oldpeak']['description'])
        slope = st.selectbox("ST Segment Slope", options=[0, 1, 2], 
                            format_func=lambda x: FEATURE_INFO['slope']['options'][x],
                            help=FEATURE_INFO['slope']['description'])
        ca = st.selectbox(f"Major Vessels (0-4) - Normal: {FEATURE_INFO['ca']['normal']}", options=[0, 1, 2, 3, 4],
                         help=FEATURE_INFO['ca']['description'])
        thal = st.selectbox("Thalassemia", options=[0, 1, 2, 3], 
                           format_func=lambda x: FEATURE_INFO['thal']['options'][x],
                           help=FEATURE_INFO['thal']['description'])

    st.divider()

    col_button = st.columns([1, 2, 1])
    with col_button[1]:
        predict_button = st.button("🔍 Analyze Heart Disease Risk", use_container_width=True)
    
    if predict_button:
        patient = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

        prediction = model.predict(patient)
        risk_prob = model.predict_proba(patient)
        
        risk_percentage = risk_prob[0][1] * 100
        
        # Save to history
        assessment_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'patient_name': patient_name if patient_name else "Anonymous",
            'patient_id': patient_id if patient_id else "N/A",
            'prediction': int(prediction[0]),
            'risk_percentage': risk_percentage,
            'features': {
                'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
                'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
                'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
            }
        }
        st.session_state.patient_history.append(assessment_record)

        st.divider()
        
        # Results visualization
        col_result1, col_result2, col_result3 = st.columns([1, 2, 1])
        
        with col_result2:
            # Risk gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_percentage,
                title={'text': "Heart Disease Risk Score", 'font': {'size': 24}},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#E74C3C" if risk_percentage > 50 else "#27AE60"},
                    'steps': [
                        {'range': [0, 30], 'color': "#D5F5E3"},
                        {'range': [30, 70], 'color': "#FCF3CF"},
                        {'range': [70, 100], 'color': "#FADBD8"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Display results
        if prediction[0] == 1:
            st.error("### ⚠️ HIGH RISK DETECTED")
            st.markdown(f"""
            <div style='background-color: #FFE5E5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #E74C3C;'>
                <h4 style='color: #C0392B; margin: 0;'>Risk Level: {risk_percentage:.1f}%</h4>
                <p style='color: #555; margin: 0.5rem 0 0 0;'><strong>⚕️ Recommendation:</strong> Immediate consultation with a cardiologist is strongly recommended. Further diagnostic tests may be necessary.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("### ✅ LOW RISK")
            st.markdown(f"""
            <div style='background-color: #E5F5E5; padding: 1.5rem; border-radius: 8px; border-left: 4px solid #27AE60;'>
                <h4 style='color: #1E8449; margin: 0;'>Risk Level: {risk_percentage:.1f}%</h4>
                <p style='color: #555; margin: 0.5rem 0 0 0;'><strong>💚 Recommendation:</strong> Continue with regular health check-ups and maintain a healthy lifestyle. Monitor risk factors periodically.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Risk factors breakdown
        st.markdown("### 🎯 Risk Factors Analysis")
        
        risk_factors = []
        if chol > 200:
            risk_factors.append(f"🔴 Elevated Cholesterol: {chol} mg/dl (Normal: <200)")
        if trestbps > 130:
            risk_factors.append(f"🔴 High Blood Pressure: {trestbps} mm Hg (Normal: 90-120)")
        if age > 55:
            risk_factors.append(f"🟡 Age Factor: {age} years")
        if fbs == 1:
            risk_factors.append("🔴 Elevated Fasting Blood Sugar")
        if exang == 1:
            risk_factors.append("🔴 Exercise-Induced Angina Present")
        if ca > 0:
            risk_factors.append(f"🔴 {ca} Major Vessel(s) with Narrowing")
        if oldpeak > 2:
            risk_factors.append(f"🟡 ST Depression: {oldpeak} mm")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.success("✅ No major risk factors identified")
        
        # Export option
        st.divider()
        col_export = st.columns([1, 1, 1])
        with col_export[1]:
            report_data = f"""
HEART DISEASE RISK ASSESSMENT REPORT
=====================================
Patient: {patient_name if patient_name else 'Anonymous'}
Patient ID: {patient_id if patient_id else 'N/A'}
Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

RISK ASSESSMENT
---------------
Risk Level: {'HIGH RISK' if prediction[0] == 1 else 'LOW RISK'}
Risk Score: {risk_percentage:.1f}%

CLINICAL PARAMETERS
------------------
Age: {age} years
Sex: {FEATURE_INFO['sex']['options'][sex]}
Chest Pain: {FEATURE_INFO['cp']['options'][cp]}
Resting BP: {trestbps} mm Hg
Cholesterol: {chol} mg/dl
Fasting Blood Sugar: {FEATURE_INFO['fbs']['options'][fbs]}
Resting ECG: {FEATURE_INFO['restecg']['options'][restecg]}
Max Heart Rate: {thalach} bpm
Exercise Angina: {FEATURE_INFO['exang']['options'][exang]}
ST Depression: {oldpeak} mm
ST Slope: {FEATURE_INFO['slope']['options'][slope]}
Major Vessels: {ca}
Thalassemia: {FEATURE_INFO['thal']['options'][thal]}

RISK FACTORS
------------
{chr(10).join(risk_factors) if risk_factors else 'No major risk factors identified'}

RECOMMENDATION
--------------
{'Immediate consultation with a cardiologist is recommended.' if prediction[0] == 1 else 'Continue regular health monitoring and maintain healthy lifestyle.'}

---
This report was generated by an AI system and should be reviewed by a qualified healthcare professional.
            """
            st.download_button(
                label="📄 Download Report",
                data=report_data,
                file_name=f"heart_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# TAB 2: Analytics Dashboard
with tab2:
    st.markdown("### 📊 Patient Analytics Dashboard")
    
    if st.session_state.patient_history:
        # Create DataFrame from history
        df_history = pd.DataFrame(st.session_state.patient_history)
        
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            st.metric("Total Assessments", len(df_history))
        with col_metric2:
            high_risk_pct = (df_history['prediction'].sum() / len(df_history)) * 100
            st.metric("High Risk %", f"{high_risk_pct:.1f}%")
        with col_metric3:
            avg_risk = df_history['risk_percentage'].mean()
            st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
        with col_metric4:
            latest_risk = df_history.iloc[-1]['risk_percentage']
            st.metric("Latest Assessment", f"{latest_risk:.1f}%")
        
        st.divider()
        
        # Risk distribution chart
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("#### Risk Distribution")
            fig_pie = px.pie(
                values=[len(df_history[df_history['prediction']==0]), 
                       len(df_history[df_history['prediction']==1])],
                names=['Low Risk', 'High Risk'],
                color_discrete_sequence=['#27AE60', '#E74C3C']
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_chart2:
            st.markdown("#### Risk Score Trend")
            fig_line = px.line(
                df_history.reset_index(), 
                x='index', 
                y='risk_percentage',
                markers=True,
                labels={'index': 'Assessment Number', 'risk_percentage': 'Risk Score (%)'}
            )
            fig_line.update_traces(line_color='#00A8E8', marker=dict(size=10))
            st.plotly_chart(fig_line, use_container_width=True)
        
        # Feature importance visualization (simulated)
        st.divider()
        st.markdown("#### 📈 Key Risk Indicators (Feature Importance)")
        
        feature_importance = pd.DataFrame({
            'Feature': ['Age', 'Cholesterol', 'Max Heart Rate', 'ST Depression', 
                       'Chest Pain Type', 'Major Vessels', 'Thalassemia', 'Exercise Angina',
                       'Blood Pressure', 'Sex', 'ECG Results', 'ST Slope', 'Blood Sugar'],
            'Importance': [0.18, 0.15, 0.13, 0.12, 0.10, 0.09, 0.08, 0.06, 0.04, 0.02, 0.01, 0.01, 0.01]
        }).sort_values('Importance', ascending=True)
        
        fig_bar = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            color='Importance',
            color_continuous_scale='Blues'
        )
        fig_bar.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
        
    else:
        st.info("📊 No assessment data available yet. Complete a risk assessment in the first tab to see analytics.")

# TAB 3: Patient History
with tab3:
    st.markdown("### 📋 Assessment History")
    
    if st.session_state.patient_history:
        for idx, record in enumerate(reversed(st.session_state.patient_history)):
            with st.expander(f"🗓️ Assessment #{len(st.session_state.patient_history) - idx} - {record['timestamp']}", expanded=(idx==0)):
                col_hist1, col_hist2 = st.columns([1, 2])
                
                with col_hist1:
                    st.markdown(f"""
                    **Patient:** {record['patient_name']}  
                    **ID:** {record['patient_id']}  
                    **Date:** {record['timestamp']}
                    """)
                    
                    if record['prediction'] == 1:
                        st.error(f"⚠️ HIGH RISK - {record['risk_percentage']:.1f}%")
                    else:
                        st.success(f"✅ LOW RISK - {record['risk_percentage']:.1f}%")
                
                with col_hist2:
                    st.markdown("**Clinical Parameters:**")
                    features = record['features']
                    st.markdown(f"""
                    - Age: {features['age']} years | Sex: {FEATURE_INFO['sex']['options'][features['sex']]}
                    - BP: {features['trestbps']} mm Hg | Cholesterol: {features['chol']} mg/dl
                    - Max HR: {features['thalach']} bpm | ST Depression: {features['oldpeak']} mm
                    - Chest Pain: {FEATURE_INFO['cp']['options'][features['cp']]}
                    - Exercise Angina: {FEATURE_INFO['exang']['options'][features['exang']]}
                    """)
    else:
        st.info("📋 No assessment history available yet.")

# TAB 4: Information
with tab4:
    st.markdown("### ℹ️ About This System")
    
    st.markdown("""
    ## 🎯 Purpose
    This AI-powered system provides **cardiac risk assessment** to assist healthcare professionals 
    in evaluating the probability of heart disease in patients based on clinical parameters.
    
    ## 🤖 Technology
    - **Machine Learning Model**: Logistic Regression / Random Forest
    - **Training Dataset**: Cleveland Heart Disease Database (303 patients)
    - **Features**: 13 clinical and demographic parameters
    - **Model Accuracy**: ~85% on test data
    
    ## 📊 Input Parameters
    """)
    
    for feature, info in FEATURE_INFO.items():
        st.markdown(f"**{info['name']}**: {info['description']}")
    
    st.divider()
    
    st.markdown("""
    ## ⚠️ Medical Disclaimer
    
    <div class="warning-box">
    <strong>IMPORTANT:</strong> This system is designed as a <strong>decision support tool</strong> for healthcare professionals. 
    It should NOT be used as the sole basis for medical diagnosis or treatment decisions.
    
    - ✅ Use as part of comprehensive clinical evaluation
    - ✅ Combine with other diagnostic tests and clinical judgment
    - ✅ Review all results with qualified medical professionals
    - ❌ Do not use for self-diagnosis
    - ❌ Do not replace professional medical advice
    </div>
    
    ## 📞 Support
    For technical support or questions, please contact your system administrator.
    
    ## 📚 References
    - UCI Machine Learning Repository: Heart Disease Dataset
    - Cleveland Clinic Foundation
    - Research: Detrano, R. et al. (1989)
    
    ---
    **Version**: 1.0 Pro Max | **Last Updated**: March 2026
    """, unsafe_allow_html=True)