import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Production UI Theme & Dashboard Config
st.set_page_config(
    page_title="FraudGuard AI // Risk Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Overrides (Glassmorphism + Cyberpunk Dark Mode)
st.markdown("""
    <style>
    /* Main body styling */
    .stApp {
        background-color: #0b0f19;
        font-family: 'Inter', system-ui, sans-serif;
    }
    
    /* Elegant Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #111625 !important;
        border-right: 1px solid #1f293d;
    }
    
    /* Metric Blocks */
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        font-weight: 700;
        color: #00f0ff !important;
        letter-spacing: -0.5px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8fa0c2 !important;
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 1px;
    }
    div[data-testid="stMetric"] {
        background: rgba(17, 24, 39, 0.5);
        border: 1px solid #1f293d;
        padding: 15px 20px;
        border-radius: 12px;
    }

    /* Glowing Title Header Banner */
    .header-banner {
        background: linear-gradient(135deg, #111827 0%, #1e1b4b 100%);
        padding: 35px;
        border-radius: 16px;
        border: 1px solid #312e81;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
        margin-bottom: 30px;
    }
    .header-title {
        color: #ffffff;
        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 0;
        background: linear-gradient(90deg, #ffffff, #a5b4fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-top: 8px;
    }

    /* Container Cards */
    .feature-card {
        background: #111827;
        padding: 24px;
        border-radius: 14px;
        border: 1px solid #1f293d;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }

    /* Styled Input Labels */
    label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }

    /* Tab UI styling styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #111625;
        border: 1px solid #1f293d;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #312e81 !important;
        color: white !important;
        border-color: #4f46e5 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Asset Loader
@st.cache_resource
def load_assets():
    with open('random_forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except FileNotFoundError:
    st.error("Assets Not Found. Run your notebook to export 'random_forest_model.pkl' and 'scaler.pkl'.")
    st.stop()

# 3. Sidebar Panel Layout
with st.sidebar:
    st.markdown("### 🛠️ System Overview")
    st.markdown("---")
    st.markdown("**Data Engineering Pipeline:**\n`RobustScaler` applied to numeric vectors.")
    st.markdown("**Core Logic Engine:**\n`RandomForestClassifier` (100 parallel ensemble estimators)")
    st.markdown("---")
    st.write("🔒 Connected to Enterprise Node secure feed.")

# 4. Header Banner Rendering
st.markdown("""
    <div class="header-banner">
        <h1 class="header-title">🛡️ FRAUDGUARD AI // Enterprise Risk Dashboard</h1>
        <div class="header-subtitle">Real-time anomalous transaction evaluation pipeline powered by an optimized Ensemble Random Forest Core.</div>
    </div>
""", unsafe_allow_html=True)

# 5. Core Tab Structures
tab_infer, tab_metrics = st.tabs(["🚀 Real-Time Screening Portal", "📊 Core Model Performance & Analytics"])

# ================= TAB 1: SCREENING PORTAL =================
with tab_infer:
    st.markdown('<div class="feature-card"><h3>🔍 Diagnostic Transaction Inputs</h3>Adjust parameters manually or feed transaction signals down to compute live telemetry risk ratings.</div>', unsafe_allow_html=True)
    
    # Beautiful Form Columns grouping
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 💵 Core Capital Metrics")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=85.00, step=5.0)
        time = st.number_input("Timestamp Offset (Seconds)", min_value=0.0, value=45000.0, step=100.0)
        
    with col2:
        st.markdown("##### 🧩 PCA Components (1-3)")
        v1 = st.slider("Component V1", -5.0, 5.0, 0.2)
        v2 = st.slider("Component V2", -5.0, 5.0, -0.5)
        v3 = st.slider("Component V3", -5.0, 5.0, 1.1)

    with col3:
        st.markdown("##### ⚡ Key Risk Drivers")
        v4 = st.slider("Component V4", -5.0, 5.0, 0.0)
        v11 = st.slider("Predictor V11 (Highly Weighted)", -5.0, 5.0, -1.8)
        v12 = st.slider("Predictor V12 (Highly Weighted)", -5.0, 5.0, 2.0)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Premium Button Action
    if st.button("⚡ Run Security Verification System", use_container_width=True):
        scaled_amount = scaler.transform([[amount]])[0][0]
        scaled_time = scaler.transform([[time]])[0][0]
        
        feature_vector = np.zeros(30)
        feature_vector[0], feature_vector[1], feature_vector[2], feature_vector[3] = v1, v2, v3, v4
        feature_vector[10], feature_vector[11] = v11, v12
        feature_vector[28] = scaled_amount
        feature_vector[29] = scaled_time
        
        fraud_prob = model.predict_proba([feature_vector])[0][1]
        
        st.markdown("### 🏁 Transaction Risk Assessment")
        # Beautifully custom-colored result cards with drop shadows
        if fraud_prob >= 0.20:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #451a1a 0%, #7f1d1d 100%); padding: 30px; border-radius: 12px; border: 1px solid #f87171; box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);">
                    <h3 style="color: #f87171; margin: 0; font-weight:700;">🚨 HIGH-RISK ANOMALY INTERCEPTED</h3>
                    <p style="color: #fca5a5; font-size: 18px; margin-top: 12px; margin-bottom: 0;">
                        The verification script flags this transaction pattern as inconsistent with standard safe accounts.<br>
                        <span style="font-size:24px; font-weight:800; color:#ffffff;">Estimated Fraud Probability: {fraud_prob * 100:.2f}%</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #064e3b 0%, #065f46 100%); padding: 30px; border-radius: 12px; border: 1px solid #34d399; box-shadow: 0 10px 15px -3px rgba(52, 211, 153, 0.2);">
                    <h3 style="color: #34d399; margin: 0; font-weight:700;">✅ TRANSACTION AUTHORIZED</h3>
                    <p style="color: #a7f3d0; font-size: 18px; margin-top: 12px; margin-bottom: 0;">
                        The verification parameters align safely within typical consumer behavior structures.<br>
                        <span style="font-size:24px; font-weight:800; color:#ffffff;">Estimated Fraud Probability: {fraud_prob * 100:.2f}%</span>
                    </p>
                </div>
            """, unsafe_allow_html=True)

# ================= TAB 2: METRICS & VISUALS =================
with tab_metrics:
    st.markdown('<div class="feature-card"><h3>📈 Model Validation Analytics</h3>Performance matrices generated across various algorithm paradigms to defend class skewness.</div>', unsafe_allow_html=True)
    
    # Structured KPIs Rows
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric(label="Selected Baseline Engine", value="Random Forest")
    kpi2.metric(label="Optimized F1-Score", value="85.00%")
    kpi3.metric(label="False Positives (~57k test)", value="3 Cases")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # High-end Custom Charts
    chart_col1, chart_col2 = st.columns(2)
    plt.style.use('dark_background')
    
    with chart_col1:
        st.markdown("#### 🎯 Skewed F1-Score Benchmark Summary")
        models = ["Logistic (Base)", "Logistic (Tuned)", "XGBoost Core", "Random Forest"]
        f1_scores = [0.11, 0.27, 0.63, 0.85]
        
        fig1, ax1 = plt.subplots(figsize=(6, 3.8))
        colors = ['#374151', '#1e3a8a', '#3b82f6', '#00f0ff'] # Sleek blues and glowing cyans
        bars = ax1.barh(models, f1_scores, color=colors, height=0.55, edgecolor='#1f293d')
        
        ax1.set_xlim(0, 1.0)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_color('#1f293d')
        ax1.spines['bottom'].set_color('#1f293d')
        
        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.2f}', 
                     va='center', ha='left', color='#ffffff', fontweight='bold', fontsize=10)
                     
        fig1.patch.set_facecolor('#111827')
        ax1.set_facecolor('#111827')
        st.pyplot(fig1)

    with chart_col2:
        st.markdown("#### ⚠️ Operational False Alarm Comparison (Log Axis)")
        false_positives = [1389, 460, 83, 3] 
        
        fig2, ax2 = plt.subplots(figsize=(6, 3.8))
        ax2.set_yscale('log')
        
        sns.barplot(x=models, y=false_positives, palette="Purples_r", ax=ax2, edgecolor='#1f293d')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_color('#1f293d')
        ax2.spines['bottom'].set_color('#1f293d')
        ax2.set_ylabel("Blocked Customers Count", color="#94a3b8")
        
        for i, val in enumerate(false_positives):
            ax2.text(i, val * 1.25 if val > 10 else val + 1, str(val), 
                     ha='center', va='bottom', color='white', fontweight='bold', fontsize=10)
                     
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')
        st.pyplot(fig2)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Modern Clean Dataframe View
    st.markdown("#### Detailed Cross-Architecture Benchmark Breakdown")
    comparison_data = {
        "Classifier Architecture": ["Logistic Regression (Baseline)", "Logistic Regression (Tuned)", "Random Forest (Default Core)", "XGBoost Engine"],
        "Precision Score": ["0.06", "0.16", "0.96", "0.50"],
        "Recall Accuracy (Sensitivity)": ["0.92", "0.89", "0.76", "0.85"],
        "Resulting F1-Score": ["0.11", "0.27", "0.85", "0.63"],
        "Customer Disruptions (False Positives)": ["1,389", "460", "3", "83"]
    }
    df_metrics = pd.DataFrame(comparison_data)
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
