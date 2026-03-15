

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os

st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main background with subtle gradient */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #ffffff;
        border-right: 2px solid #dee2e6;
        box-shadow: 2px 0 10px rgba(0,0,0,0.05);
    }
    
    /* Headers */
    h1 {
        color: #1a2b4c;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    h2 {
        color: #2c3e50;
        font-weight: 600;
        font-size: 2rem;
    }
    h3 {
        color: #34495e;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    /* Metric cards with glassmorphism effect */
    .metric-card {
        background: rgba(255,255,255,0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(255,255,255,0.5);
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    .metric-card h3 {
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    .metric-card h2 {
        color: #1a2b4c;
        font-size: 2.5rem;
        margin: 10px 0;
    }
    .metric-card p {
        color: #6c757d;
        font-size: 0.9rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 12px 30px;
        font-size: 1.1rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    /* Prediction result boxes */
    .result-box-success {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 8px solid #28a745;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(40, 167, 69, 0.2);
    }
    .result-box-error {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 8px solid #dc3545;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(220, 53, 69, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(255,255,255,0.5);
        padding: 10px;
        border-radius: 50px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 50px;
        padding: 10px 25px;
        background: transparent;
        color: #495057;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        box-shadow: 0 4px 10px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar inputs */
    .stSlider label, .stSelectbox label {
        font-weight: 600;
        color: #495057;
    }
    .stSlider div[data-baseweb="slider"] {
        margin-top: 5px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 30px;
        color: #6c757d;
        font-size: 0.9rem;
        border-top: 1px solid #dee2e6;
        margin-top: 50px;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.pkl')
    rf_model = joblib.load('random_forest.pkl')
    lr_model = joblib.load('logistic_regression.pkl')
    xgb_model = joblib.load('xgboost.pkl')
    return scaler, rf_model, lr_model, xgb_model

@st.cache_data
def load_results():
    return pd.read_csv('model_results.csv', index_col=0)

scaler, rf_model, lr_model, xgb_model = load_models()
results_df = load_results()

# ---------------------------
# Sidebar - Patient Input
# ---------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/heart-with-pulse.png", width=80)
    st.title("Patient Data")
    st.markdown("---")
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 20, 100, 50, help="Years")
        sex = st.selectbox("Sex", [("Female", 0), ("Male", 1)], format_func=lambda x: x[0])[1]
        cp = st.selectbox("Chest Pain", 
                          [("Typical angina", 1), ("Atypical angina", 2), 
                           ("Non-anginal", 3), ("Asymptomatic", 4)],
                          format_func=lambda x: x[0])[1]
        bp = st.slider("BP (mm Hg)", 90, 200, 130)
        chol = st.slider("Cholesterol", 100, 600, 250)
    with col2:
        fbs = st.selectbox("FBS > 120", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        ekg = st.selectbox("ECG", [("Normal", 0), ("ST-T wave", 1), ("LVH", 2)], format_func=lambda x: x[0])[1]
        max_hr = st.slider("Max HR", 70, 210, 150)
        exang = st.selectbox("Exercise Angina", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
    
    # Second row
    col3, col4 = st.columns(2)
    with col3:
        slope = st.selectbox("ST Slope", [("Upsloping", 1), ("Flat", 2), ("Downsloping", 3)], format_func=lambda x: x[0])[1]
    with col4:
        ca = st.selectbox("Vessels", [0,1,2,3])
    
    thal = st.selectbox("Thallium", [("Normal", 3), ("Fixed defect", 6), ("Reversible", 7)], format_func=lambda x: x[0])[1]
    
    st.markdown("---")
    st.caption("🔍 Adjust values and click a model to predict.")

# Prepare input array
input_data = np.array([[age, sex, cp, bp, chol, fbs, ekg, max_hr, exang, oldpeak, slope, ca, thal]])
input_scaled = scaler.transform(input_data)

# ---------------------------
# Main area
# ---------------------------
# Header with icon
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
with col_title:
    st.title("Heart Disease Risk Assessment")
    st.markdown("#### *Powered by Machine Learning*")

# Metric cards
st.markdown("### 📊 Overview")
col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
with col_metrics1:
    st.markdown('<div class="metric-card"><h3>🤖 Models</h3><h2>3</h2><p>Random Forest, Logistic Regression, XGBoost</p></div>', unsafe_allow_html=True)
with col_metrics2:
    st.markdown('<div class="metric-card"><h3>📋 Features</h3><h2>13</h2><p>Clinical parameters</p></div>', unsafe_allow_html=True)
with col_metrics3:
    st.markdown('<div class="metric-card"><h3>🎯 Avg. Accuracy</h3><h2>85%</h2><p>Across all models</p></div>', unsafe_allow_html=True)

st.markdown("---")

# Prediction section
st.markdown("### 🔮 Predict Heart Disease")
st.markdown("Choose a model to get a prediction based on the patient data entered.")

col_btn1, col_btn2, col_btn3 = st.columns(3)
with col_btn1:
    if st.button("🌲 Random Forest", use_container_width=True):
        pred = rf_model.predict(input_scaled)[0]
        prob = rf_model.predict_proba(input_scaled)[0]
        st.session_state['pred'] = pred
        st.session_state['prob'] = prob
        st.session_state['model'] = 'Random Forest'

with col_btn2:
    if st.button("📈 Logistic Regression", use_container_width=True):
        pred = lr_model.predict(input_scaled)[0]
        prob = lr_model.predict_proba(input_scaled)[0]
        st.session_state['pred'] = pred
        st.session_state['prob'] = prob
        st.session_state['model'] = 'Logistic Regression'

with col_btn3:
    if st.button("⚡ XGBoost", use_container_width=True):
        pred = xgb_model.predict(input_scaled)[0]
        prob = xgb_model.predict_proba(input_scaled)[0]
        st.session_state['pred'] = pred
        st.session_state['prob'] = prob
        st.session_state['model'] = 'XGBoost'

# Display prediction result
if 'pred' in st.session_state:
    pred_label = "Presence" if st.session_state['pred'] == 1 else "Absence"
    prob_value = st.session_state['prob'][1] if st.session_state['pred'] == 1 else st.session_state['prob'][0]
    
    # Color-coded result box
    if st.session_state['pred'] == 1:
        st.markdown(f"""
        <div class="result-box-error">
            <h3 style="color:#721c24; margin:0;">❌ High Risk: {pred_label}</h3>
            <p style="font-size:1.5rem; color:#721c24; margin:5px 0;">Probability: {prob_value:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-box-success">
            <h3 style="color:#155724; margin:0;">✅ Low Risk: {pred_label}</h3>
            <p style="font-size:1.5rem; color:#155724; margin:5px 0;">Probability: {prob_value:.2%}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_value * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Probability (%)", 'font': {'size': 24, 'color': '#2c3e50'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': '#2c3e50'},
            'bar': {'color': "#dc3545" if prob_value > 0.5 else "#28a745"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "#ced4da",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}],
            'threshold': {
                'line': {'color': '#1a2b4c', 'width': 4},
                'thickness': 0.75,
                'value': 50}}))
    fig.update_layout(height=300, margin=dict(l=30, r=30, t=70, b=30))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ---------------------------
# Visualizations tabs
# ---------------------------
st.markdown("### 📊 Model Performance & Explainability")

tab1, tab2, tab3 = st.tabs(["📈 Model Comparison", "🔲 Confusion Matrices", "🔍 SHAP Explanation"])

with tab1:
    st.markdown("#### Comparison of Evaluation Metrics")
    fig = px.bar(results_df, barmode='group', 
                 labels={'value': 'Score', 'index': 'Model'},
                 color_discrete_sequence=px.colors.sequential.Plasma)
    fig.update_layout(legend_title_text='Metric', xaxis_title='', yaxis_title='Score',
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                      font=dict(color='#2c3e50'))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.markdown("#### Confusion Matrices")
    if os.path.exists('confusion_matrices.png'):
        image = Image.open('confusion_matrices.png')
        st.image(image, caption='Confusion Matrices', use_column_width=True)
    else:
        st.warning("Confusion matrix image not found. Please run model.py first.")

with tab3:
    st.markdown("#### Feature Importance (SHAP)")
    st.markdown("SHAP values show the impact of each feature on the prediction.")
    if os.path.exists('shap_summary.png'):
        image = Image.open('shap_summary.png')
        st.image(image, caption='SHAP Summary Plot', use_column_width=True)
    else:
        st.warning("SHAP plot not found. Please run model.py first.")

# ---------------------------
# Footer
# ---------------------------
st.markdown('<div class="footer">© 2026 Heart Disease Research Project | CSE-AIML | Built with ❤️ using Streamlit</div>', unsafe_allow_html=True)
