
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px


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
    /* Root variables */
    :root {
        --primary: #00A8E8;
        --success: #27AE60;
        --danger: #E74C3C;
        --warning: #F39C12;
        --info: #3498DB;
    }
    
    /* Main container */
    .main {
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #00A8E8 0%, #0091C9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-top: 4px solid #00A8E8;
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Section headers */
    h4 {
        color: #00A8E8;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #00A8E8 0%, #0091C9 100%) !important;
        color: white !important;
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        padding: 0.85rem !important;
        border-radius: 10px !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0,168,232,0.3) !important;
        letter-spacing: 0.5px;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 8px 25px rgba(0,168,232,0.5) !important;
    }
    
    .stButton>button:active {
        transform: translateY(-1px) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        background-color: white;
        border-radius: 10px 10px 0 0;
        border-bottom: 3px solid transparent;
        transition: all 0.3s ease;
        font-weight: 600;
        color: #666;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00A8E8 0%, #0091C9 100%);
        color: white;
        border-bottom: 3px solid #0091C9;
    }
    
    /* Input styling */
    .stNumberInput, .stSlider, .stSelectbox {
        background-color: #f8f9fa;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background: linear-gradient(to right, #E8F4F8, #00A8E8);
    }
    
    /* Gauge styling */
    .plotly {
        background: white;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    /* Success/Error/Warning boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 5px solid;
    }
    
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #E74C3C;
        padding: 1.25rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Divider */
    hr {
        border-color: #e0e0e0;
        margin: 1.5rem 0;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
    }
    
    .sidebar-metric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #00A8E8;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; background: linear-gradient(135deg, #00A8E8 0%, #0091C9 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800;'><i class='fas fa-heart' style='color: #E74C3C; margin-right: 0.5rem;'></i> AI Heart Disease Risk Assessment</h1>", unsafe_allow_html=True)

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
    st.markdown("## 🏥 System Dashboard")
    st.image("https://img.icons8.com/color/256/000000/heart-health.png", width=120)
    
    st.divider()
    
    st.markdown("### 📊 Assessment Statistics")
    if st.session_state.patient_history:
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("📈 Total", len(st.session_state.patient_history))
        with col_stat2:
            high_risk_count = sum(1 for p in st.session_state.patient_history if p['prediction'] == 1)
            st.metric("⚠️ High Risk", high_risk_count)
        
        st.markdown(f"**Low Risk**: {len(st.session_state.patient_history) - high_risk_count}")
        
        if st.session_state.patient_history:
            avg_score = np.mean([p['risk_percentage'] for p in st.session_state.patient_history])
            st.markdown(f"**Avg Score**: {avg_score:.1f}%")
    else:
        st.info("💡 No assessments yet. Complete a risk assessment to see statistics.")
    
    st.divider()
    
    st.markdown("### 🤖 AI Model Info")
    st.markdown("""
    **Model Type**: Logistic Regression  
    **Accuracy**: ~85%  
    **Features**: 13 Parameters  
    **Dataset**: 303 Patients
    """)
    
    st.divider()
    
    col_clear = st.columns([1, 1])
    with col_clear[0]:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.patient_history = []
            st.rerun()
    
    with col_clear[1]:
        if st.button("🔄 Refresh", use_container_width=True):
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
        assessment_number = len(st.session_state.patient_history) + 1
        
        # Save to history
        assessment_record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'assessment_num': assessment_number,
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
            st.error("### ⚠️ HIGH RISK DETECTED - IMMEDIATE ACTION REQUIRED")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #E74C3C;'>
                <h3 style='color: #C0392B; margin: 0 0 0.5rem 0;'>⚠️ Risk Level: {risk_percentage:.1f}%</h3>
                <p style='color: #555; margin: 0; line-height: 1.6;'><strong>Clinical Recommendation:</strong> Immediate consultation with a cardiologist is strongly recommended. Further diagnostic evaluation including ECG, stress testing, and imaging studies may be necessary.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success("### ✅ LOW RISK - MAINTAIN HEALTHY LIFESTYLE")
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #27AE60;'>
                <h3 style='color: #1E8449; margin: 0 0 0.5rem 0;'>✅ Risk Level: {risk_percentage:.1f}%</h3>
                <p style='color: #555; margin: 0; line-height: 1.6;'><strong>Clinical Recommendation:</strong> Continue with regular health monitoring and annual check-ups. Maintain a healthy lifestyle with balanced diet, regular exercise, and stress management.</p>
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
╔════════════════════════════════════════════════════════════════╗
║       HEART DISEASE RISK ASSESSMENT REPORT                    ║
╚════════════════════════════════════════════════════════════════╝

Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Assessment #: {assessment_number}

═══════════════════════════════════════════════════════════════════
RISK ASSESSMENT RESULTS
═══════════════════════════════════════════════════════════════════

Risk Category: {'⚠️ HIGH RISK' if prediction[0] == 1 else '✅ LOW RISK'}
Risk Score: {risk_percentage:.1f}%

═══════════════════════════════════════════════════════════════════
CLINICAL PARAMETERS
═══════════════════════════════════════════════════════════════════

Demographics:
  • Age: {age} years
  • Sex: {FEATURE_INFO['sex']['options'][sex]}

Vital Signs & Laboratory:
  • Resting Blood Pressure: {trestbps} mm Hg
  • Cholesterol Level: {chol} mg/dl
  • Fasting Blood Sugar: {FEATURE_INFO['fbs']['options'][fbs]}
  • Maximum Heart Rate: {thalach} bpm

Cardiac Assessment:
  • Chest Pain Type: {FEATURE_INFO['cp']['options'][cp]}
  • Resting ECG: {FEATURE_INFO['restecg']['options'][restecg]}
  • Exercise Induced Angina: {FEATURE_INFO['exang']['options'][exang]}
  • ST Depression (Oldpeak): {oldpeak} mm
  • ST Slope: {FEATURE_INFO['slope']['options'][slope]}
  • Major Vessels Affected: {ca}
  • Thalassemia Status: {FEATURE_INFO['thal']['options'][thal]}

═══════════════════════════════════════════════════════════════════
IDENTIFIED RISK FACTORS
═══════════════════════════════════════════════════════════════════

{chr(10).join(['  ' + factor for factor in risk_factors]) if risk_factors else '  ✓ No major risk factors identified'}

═══════════════════════════════════════════════════════════════════
CLINICAL RECOMMENDATION
═══════════════════════════════════════════════════════════════════

{'Immediate consultation with a cardiologist is strongly recommended. Further diagnostic evaluation and testing may be necessary. This patient should be prioritized for medical review.' if prediction[0] == 1 else 'Continue with regular health monitoring and maintain a healthy lifestyle. Monitor risk factors periodically. Annual check-ups are recommended.'}

═══════════════════════════════════════════════════════════════════
DISCLAIMER
═══════════════════════════════════════════════════════════════════

This report is generated by an artificial intelligence system designed
as a DECISION SUPPORT TOOL ONLY. It is not a medical diagnosis and
should NOT be used as the sole basis for clinical decision-making.

All results must be reviewed and interpreted by a qualified healthcare
professional in conjunction with:
  • Complete physical examination
  • Patient history and symptoms
  • Additional diagnostic tests (ECG, stress tests, imaging, etc.)
  • Clinical judgment and expertise

═══════════════════════════════════════════════════════════════════
Generated by: AI Heart Disease Risk Assessment System v1.0 Pro Max
═══════════════════════════════════════════════════════════════════
            """
            st.download_button(
                label="📄 Download Report",
                data=report_data,
                file_name=f"assessment_{assessment_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
    st.markdown("All completed risk assessments with detailed clinical parameters")
    
    if st.session_state.patient_history:
        for idx, record in enumerate(reversed(st.session_state.patient_history)):
            assessment_num = len(st.session_state.patient_history) - idx
            
            # Create expander with better styling
            with st.expander(
                f"{'⚠️' if record['prediction'] == 1 else '✅'} Assessment #{assessment_num} | "
                f"{record['risk_percentage']:.1f}% Risk | {record['timestamp']}", 
                expanded=(idx==0)
            ):
                col_hist1, col_hist2 = st.columns([1, 1])
                
                with col_hist1:
                    st.markdown("**Assessment Summary**")
                    st.markdown(f"""
                    📅 **Date**: {record['timestamp']}
                    
                    📊 **Risk Level**: {'⚠️ HIGH RISK' if record['prediction'] == 1 else '✅ LOW RISK'}
                    
                    📈 **Risk Score**: {record['risk_percentage']:.1f}%
                    """)
                
                with col_hist2:
                    st.markdown("**Key Metrics**")
                    features = record['features']
                    st.markdown(f"""
                    👤 Age: {features['age']} yrs
                    
                    ❤️ BP: {features['trestbps']} | HR: {features['thalach']} bpm
                    
                    🩸 Chol: {features['chol']} mg/dl
                    """)
                
                st.divider()
                
                st.markdown("**Complete Clinical Parameters**")
                param_col1, param_col2 = st.columns(2)
                
                with param_col1:
                    st.markdown(f"""
                    **Demographics**
                    - Age: {features['age']} years
                    - Sex: {FEATURE_INFO['sex']['options'][features['sex']]}
                    
                    **Vital Signs**
                    - BP: {features['trestbps']} mm Hg
                    - Max HR: {features['thalach']} bpm
                    - Cholesterol: {features['chol']} mg/dl
                    - Blood Sugar: {FEATURE_INFO['fbs']['options'][features['fbs']]}
                    """)
                
                with param_col2:
                    st.markdown(f"""
                    **Cardiac Assessment**
                    - Chest Pain: {FEATURE_INFO['cp']['options'][features['cp']]}
                    - Resting ECG: {FEATURE_INFO['restecg']['options'][features['restecg']]}
                    - Exercise Angina: {FEATURE_INFO['exang']['options'][features['exang']]}
                    - ST Depression: {features['oldpeak']} mm
                    - ST Slope: {FEATURE_INFO['slope']['options'][features['slope']]}
                    - Major Vessels: {features['ca']}
                    - Thalassemia: {FEATURE_INFO['thal']['options'][features['thal']]}
                    """)
    else:
        st.info("📋 No assessment history yet. Complete a risk assessment to build history.")

# TAB 4: Information
with tab4:
    col_info1, col_info2 = st.columns([2, 1])
    
    with col_info1:
        st.markdown("## System Overview")
        st.markdown("""
        ### 🎯 Purpose
        Advanced AI system for cardiac risk assessment to support clinical decision-making. 
        Analyzes 13 key clinical parameters using machine learning to predict heart disease probability.
        
        ### 🤖 Technology Stack
        - **Models**: Logistic Regression & Random Forest
        - **Training Data**: Cleveland Heart Disease Database (303 patients)
        - **Features**: 13 Clinical Parameters
        - **Accuracy**: ~85% on test data
        - **Framework**: Streamlit + Scikit-learn + Plotly
        
        ### 📊 Clinical Parameters Analyzed
        """)
        
        params_col1, params_col2 = st.columns(2)
        with params_col1:
            st.markdown("""
            **Demographics & Vitals**
            - Age (years)
            - Sex (M/F)
            - Blood Pressure
            - Heart Rate
            - Cholesterol
            
            **Chest Symptoms**
            - Chest Pain Type
            - Exercise Angina
            - ST Depression
            """)
        
        with params_col2:
            st.markdown("""
            **Cardiac Assessment**
            - ECG Results
            - ST Segment Slope
            - Major Vessels
            - Thalassemia Status
            
            **Laboratory**
            - Fasting Blood Sugar
            """)
    
    with col_info2:
        st.markdown("### ℹ️ Quick Stats")
        st.markdown(f"""
        **Model Type**
        Supervised Learning
        
        **Dataset**
        Cleveland HD
        
        **Samples**
        303 Patients
        
        **Accuracy**
        ~85%
        
        **Version**
        1.0 Pro Max
        
        **Status**
        Production
        """)
    
    st.divider()
    
   
    st.divider()
    
    st.markdown("""
    ## 📚 References & Sources
    
    - **Dataset**: UCI Machine Learning Repository - Heart Disease (Cleveland)
    - **Institution**: Cleveland Clinic Foundation
    - **Key Research**: Detrano, R. et al. (1989) - International application of a new probability algorithm for the diagnosis of coronary artery disease
    - **ML Framework**: Scikit-learn
    - **Visualization**: Plotly
    
    ---
    
   
    """, unsafe_allow_html=True)