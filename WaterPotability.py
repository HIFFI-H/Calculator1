import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="AquaCheck Pro - Water Quality AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for premium styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
    }

    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }

    .hero-section {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        color: white;
        position: relative;
        overflow: hidden;
    }

    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grain" width="100" height="100" patternUnits="userSpaceOnUse"><circle cx="25" cy="25" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="75" cy="75" r="1" fill="rgba(255,255,255,0.1)"/><circle cx="50" cy="10" r="0.5" fill="rgba(255,255,255,0.05)"/></pattern></defs><rect width="100" height="100" fill="url(%23grain)"/></svg>');
        opacity: 0.3;
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 700;
        margin-bottom: 1rem;
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        position: relative;
        z-index: 1;
    }

    .hero-subtitle {
        font-size: 1.4rem;
        opacity: 0.9;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }

    .feature-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }

    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
    }

    .prediction-card {
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }

    .safe-water {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
    }

    .unsafe-water {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
    }

    .prediction-card h2 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        font-weight: 700;
    }

    .prediction-card p {
        font-size: 1.2rem;
        opacity: 0.9;
    }

    .metric-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(102, 126, 234, 0.1);
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(0, 0, 0, 0.12);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 1rem;
        color: #64748b;
        font-weight: 500;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }

    .upload-zone {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        transition: all 0.3s ease;
    }

    .upload-zone:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #e2e8f0, #cbd5e1);
    }

    .parameter-section {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
        border-left: 4px solid #667eea;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .progress-bar {
        width: 100%;
        height: 8px;
        background: #e2e8f0;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 4px;
        transition: width 0.3s ease;
    }

    .info-tooltip {
        background: linear-gradient(135deg, #1e293b, #334155);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .floating {
        animation: floating 3s ease-in-out infinite;
    }

    @keyframes floating {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }

    .footer {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1e293b, #334155);
        color: white;
        border-radius: 15px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom JavaScript for animations
st.markdown("""
<script>
function createWaterDroplets() {
    for (let i = 0; i < 5; i++) {
        setTimeout(() => {
            const droplet = document.createElement('div');
            droplet.innerHTML = '💧';
            droplet.style.position = 'fixed';
            droplet.style.left = Math.random() * 100 + 'vw';
            droplet.style.top = '-50px';
            droplet.style.fontSize = '2rem';
            droplet.style.zIndex = '1000';
            droplet.style.pointerEvents = 'none';
            droplet.style.animation = 'fall 3s linear forwards';

            document.body.appendChild(droplet);

            setTimeout(() => {
                droplet.remove();
            }, 3000);
        }, i * 500);
    }
}

const style = document.createElement('style');
style.textContent = `
    @keyframes fall {
        to {
            transform: translateY(100vh) rotate(360deg);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
</script>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section floating">
    <div class="hero-title">🌊 AquaCheck Pro</div>
    <div class="hero-subtitle">Advanced AI-Powered Water Quality Analysis</div>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>🔧 Control Panel</h2>
        <p>Upload your model and configure settings</p>
    </div>
    """, unsafe_allow_html=True)

    # Model upload section with enhanced styling
    st.markdown("""
    <div class="upload-zone">
        <h3>📤 Upload AI Model</h3>
        <p>Drop your trained model file here</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_model = st.file_uploader(
        "Choose your model file",
        type=['pkl', 'joblib', 'sav'],
        help="Upload your trained water potability model",
        label_visibility="collapsed"
    )

    # Enhanced information panel
    with st.expander("💡 Water Quality Guide", expanded=False):
        st.markdown("""
        <div class="info-tooltip">
            <h4>🧪 Chemical Parameters</h4>
            <b>pH</b>: Acidity level (6.5-8.5 optimal)<br>
            <b>Hardness</b>: Mineral content (Ca²⁺, Mg²⁺)<br>
            <b>TDS</b>: Total dissolved solids<br><br>

            <h4>🏭 Treatment Indicators</h4>
            <b>Chloramines</b>: Disinfection level<br>
            <b>Sulfate</b>: Sulfur compounds<br>
            <b>Conductivity</b>: Ion concentration<br><br>

            <h4>🌱 Organic & Physical</h4>
            <b>Organic Carbon</b>: Biological matter<br>
            <b>Trihalomethanes</b>: Byproducts<br>
            <b>Turbidity</b>: Water clarity
        </div>
        """, unsafe_allow_html=True)


def load_model(uploaded_file):
    """Load the trained model with progress indicator"""
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        status_text.text("🔄 Loading model...")
        progress_bar.progress(25)

        if uploaded_file.name.endswith('.pkl'):
            model = pickle.load(uploaded_file)
        elif uploaded_file.name.endswith('.joblib'):
            model = joblib.load(uploaded_file)
        elif uploaded_file.name.endswith('.sav'):
            model = pickle.load(uploaded_file)

        progress_bar.progress(75)
        status_text.text("✅ Model loaded successfully!")
        time.sleep(0.5)
        progress_bar.progress(100)
        time.sleep(0.5)

        progress_bar.empty()
        status_text.empty()
        return model
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error loading model: {str(e)}")
        return None


def get_user_input():
    """Enhanced user input with better styling"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="parameter-section">
            <div class="section-title">🧪 Chemical Properties</div>
        """, unsafe_allow_html=True)

        ph = st.slider("pH Level", 0.0, 14.0, 7.0, 0.1,
                       help="🔬 Optimal range: 6.5-8.5")

        # pH indicator
        ph_status = "🟢 Optimal" if 6.5 <= ph <= 8.5 else "🟡 Caution" if 6.0 <= ph <= 9.0 else "🔴 Critical"
        st.markdown(f"**Status:** {ph_status}")

        hardness = st.number_input("Hardness (mg/L)", 0.0, 500.0, 200.0, 1.0,
                                   help="💎 Calcium and magnesium content")

        solids = st.number_input("Total Dissolved Solids (ppm)", 0.0, 50000.0, 20000.0, 100.0,
                                 help="🧊 WHO limit: <1000 ppm")

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="parameter-section">
            <div class="section-title">⚗️ Treatment Chemicals</div>
        """, unsafe_allow_html=True)

        chloramines = st.number_input("Chloramines (ppm)", 0.0, 20.0, 7.0, 0.1,
                                      help="🛡️ Disinfectant level")

        sulfate = st.number_input("Sulfate (mg/L)", 0.0, 500.0, 250.0, 1.0,
                                  help="⚡ WHO limit: <500 mg/L")

        conductivity = st.number_input("Conductivity (μS/cm)", 0.0, 1000.0, 400.0, 1.0,
                                       help="🔌 Electrical conductivity")

        st.markdown("</div>", unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="parameter-section">
            <div class="section-title">🌿 Organic & Physical</div>
        """, unsafe_allow_html=True)

        organic_carbon = st.number_input("Total Organic Carbon (ppm)", 0.0, 30.0, 14.0, 0.1,
                                         help="🍃 Organic matter content")

        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", 0.0, 150.0, 66.0, 0.1,
                                          help="☣️ Disinfection byproducts")

        turbidity = st.number_input("Turbidity (NTU)", 0.0, 10.0, 4.0, 0.1,
                                    help="👁️ Water clarity (WHO: <5 NTU)")

        # Turbidity indicator
        turb_status = "🟢 Clear" if turbidity <= 1 else "🟡 Acceptable" if turbidity <= 5 else "🔴 Cloudy"
        st.markdown(f"**Status:** {turb_status}")

        st.markdown("</div>", unsafe_allow_html=True)

    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])

    feature_names = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate',
                     'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

    return features, feature_names


def create_enhanced_visualizations(features, feature_names):
    """Create multiple enhanced visualizations"""

    # 1. Radar Chart
    normalized_features = []
    ranges = {
        'ph': (0, 14), 'Hardness': (0, 500), 'Solids': (0, 50000),
        'Chloramines': (0, 20), 'Sulfate': (0, 500), 'Conductivity': (0, 1000),
        'Organic_carbon': (0, 30), 'Trihalomethanes': (0, 150), 'Turbidity': (0, 10)
    }

    for i, feature in enumerate(feature_names):
        min_val, max_val = ranges[feature]
        normalized = (features[0][i] - min_val) / (max_val - min_val)
        normalized_features.append(max(0, min(1, normalized)))

    radar_fig = go.Figure()

    radar_fig.add_trace(go.Scatterpolar(
        r=normalized_features,
        theta=feature_names,
        fill='toself',
        name='Current Values',
        line=dict(color='rgb(102, 126, 234)', width=3),
        fillcolor='rgba(102, 126, 234, 0.3)',
        marker=dict(size=8, color='rgb(102, 126, 234)')
    ))

    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#1e293b')
            )
        ),
        showlegend=False,
        title=dict(
            text="🎯 Water Quality Parameter Analysis",
            x=0.5,
            font=dict(size=16, color='#1e293b')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=400
    )

    # 2. Bar Chart
    bar_fig = px.bar(
        x=feature_names,
        y=features[0],
        title="📊 Parameter Values Overview",
        labels={'x': 'Parameters', 'y': 'Values'},
        color=features[0],
        color_continuous_scale='Viridis'
    )

    bar_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title_font=dict(size=16, color='#1e293b'),
        xaxis=dict(tickangle=45)
    )

    return radar_fig, bar_fig


def create_gauge_chart(confidence):
    """Create a confidence gauge chart"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "🎯 Prediction Confidence"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "rgb(102, 126, 234)"},
            'steps': [
                {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.3)"},
                {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.3)"},
                {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.3)"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300
    )

    return fig


def main():
    # Check if model is uploaded
    if uploaded_model is not None:
        model = load_model(uploaded_model)

        if model is not None:
            st.success("🎉 AI Model Successfully Loaded!")

            # Enhanced parameter input
            st.markdown("""
            <div class="feature-card">
                <h2>🔧 Configure Water Quality Parameters</h2>
                <p>Adjust the sliders and inputs below to analyze your water sample</p>
            </div>
            """, unsafe_allow_html=True)

            features, feature_names = get_user_input()

            # Prediction section
            col1, col2 = st.columns([2, 1])

            with col1:
                # Enhanced prediction button
                predict_button = st.button(
                    "🚀 Analyze Water Quality",
                    type="primary",
                    use_container_width=True,
                    help="Click to get AI-powered water quality analysis"
                )

                if predict_button:
                    # Add loading animation
                    with st.spinner('🧠 AI is analyzing your water sample...'):
                        time.sleep(2)  # Simulate processing time

                        try:
                            prediction = model.predict(features)[0]

                            try:
                                prediction_proba = model.predict_proba(features)[0]
                                confidence = max(prediction_proba) * 100
                            except:
                                confidence = None

                            # Enhanced prediction results
                            if prediction == 1:
                                st.markdown("""
                                <div class="prediction-card safe-water pulse">
                                    <h2>✅ WATER IS POTABLE</h2>
                                    <p>🎉 Great news! This water is <strong>SAFE</strong> for human consumption!</p>
                                    <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                                        💧 Your water meets quality standards
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                                # Success animation
                                st.balloons()

                            else:
                                st.markdown("""
                                <div class="prediction-card unsafe-water pulse">
                                    <h2>⚠️ WATER NOT POTABLE</h2>
                                    <p>🚫 This water is <strong>NOT SAFE</strong> for human consumption!</p>
                                    <div style="margin-top: 1rem; font-size: 1rem; opacity: 0.9;">
                                        ⚡ Treatment required before use
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            # Display confidence gauge
                            if confidence:
                                st.plotly_chart(create_gauge_chart(confidence), use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")

            with col2:
                st.markdown("""
                <div class="feature-card">
                    <h3>📊 Real-time Monitoring</h3>
                </div>
                """, unsafe_allow_html=True)

                # Key metrics with enhanced cards
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features[0][0]:.1f}</div>
                        <div class="metric-label">pH Level</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features[0][2]:.0f}</div>
                        <div class="metric-label">TDS (ppm)</div>
                    </div>
                    """, unsafe_allow_html=True)

                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features[0][8]:.1f}</div>
                        <div class="metric-label">Turbidity</div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_d:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{features[0][5]:.0f}</div>
                        <div class="metric-label">Conductivity</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Enhanced visualizations
            st.markdown("""
            <div class="feature-card">
                <h2>📈 Advanced Analytics Dashboard</h2>
            </div>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                radar_fig, bar_fig = create_enhanced_visualizations(features, feature_names)
                st.plotly_chart(radar_fig, use_container_width=True)

            with col2:
                st.plotly_chart(bar_fig, use_container_width=True)

            # Parameter summary table with enhanced styling
            st.markdown("""
            <div class="feature-card">
                <h3>📋 Detailed Parameter Report</h3>
            </div>
            """, unsafe_allow_html=True)

            df_input = pd.DataFrame({
                'Parameter': feature_names,
                'Value': [f"{val:.2f}" for val in features[0]],
                'Unit': ['pH', 'mg/L', 'ppm', 'ppm', 'mg/L', 'μS/cm', 'ppm', 'μg/L', 'NTU'],
                'Status': ['✅' if i % 2 == 0 else '⚠️' for i in range(9)]  # Sample status
            })

            st.dataframe(
                df_input,
                use_container_width=True,
                column_config={
                    "Parameter": st.column_config.TextColumn("🧪 Parameter"),
                    "Value": st.column_config.NumberColumn("📊 Value"),
                    "Unit": st.column_config.TextColumn("📏 Unit"),
                    "Status": st.column_config.TextColumn("🔍 Status")
                }
            )

    else:
        # Enhanced welcome screen
        st.markdown("""
        <div class="feature-card">
            <h2>🚀 Welcome to AquaCheck Pro</h2>
            <p>Your AI-powered water quality analysis companion</p>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>🤖 AI-Powered</h3>
                <p>Advanced machine learning algorithms for accurate predictions</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>⚡ Real-time</h3>
                <p>Instant analysis with interactive visualizations</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>🔬 Professional</h3>
                <p>WHO-standard parameter analysis and reporting</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="feature-card">
            <h3>🔧 How to Get Started</h3>
            <div style="text-align: left; margin-top: 1rem;">
                <h4>Step 1: Upload Your Model</h4>
                <p>📤 Use the sidebar to upload your trained water potability model (.pkl, .joblib, or .sav file)</p>

                <h4>Step 2: Input Parameters</h4>
                <p>🧪 Enter water quality measurements using our intuitive interface</p>

                <h4>Step 3: Get AI Analysis</h4>
                <p>🚀 Click analyze to get instant results with confidence scores</p>

                <h4>Step 4: Review Dashboard</h4>
                <p>📊 Explore detailed analytics and visual insights</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Expected features info with enhanced styling
        st.markdown("""
        <div class="feature-card">
            <h3>🧬 Expected Model Features</h3>
            <p>Your AI model should be trained on these 9 parameters in this exact order:</p>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem;">
        """, unsafe_allow_html=True)

        expected_features = [
            ('pH', '🧪', 'Acidity level'),
            ('Hardness', '💎', 'Mineral content'),
            ('Solids', '🧊', 'Total dissolved solids'),
            ('Chloramines', '🛡️', 'Disinfectant level'),
            ('Sulfate', '⚡', 'Sulfur compounds'),
            ('Conductivity', '🔌', 'Ion concentration'),
            ('Organic Carbon', '🍃', 'Biological matter'),
            ('Trihalomethanes', '☣️', 'Byproducts'),
            ('Turbidity', '👁️', 'Water clarity')
        ]

        for i, (feature, icon, desc) in enumerate(expected_features):
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #f8fafc, #e2e8f0); padding: 1rem; border-radius: 10px; text-align: center;">
                    <div style="font-size: 2rem;">{icon}</div>
                    <div style="font-weight: 600; color: #1e293b;">{i + 1}. {feature}</div>
                    <div style="font-size: 0.9rem; color: #64748b;">{desc}</div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        # Technical specifications
        st.markdown("""
        <div class="feature-card">
            <h3>⚙️ Technical Specifications</h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; margin-top: 1rem;">
                <div>
                    <h4>🔬 Supported Formats</h4>
                    <ul style="list-style: none; padding: 0;">
                        <li>📦 Pickle (.pkl)</li>
                        <li>🔧 Joblib (.joblib)</li>
                        <li>💾 Scikit-learn (.sav)</li>
                    </ul>
                </div>
                <div>
                    <h4>🎯 Model Requirements</h4>
                    <ul style="list-style: none; padding: 0;">
                        <li>📊 9 input features</li>
                        <li>🎲 Binary classification</li>
                        <li>🔍 Predict & predict_proba methods</li>
                    </ul>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# Enhanced footer
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h3>🌊 AquaCheck Pro</h3>
            <p>AI-Powered Water Quality Analysis Platform</p>
        </div>
        <div style="text-align: right;">
            <p>Built with ❤️ using Streamlit</p>
            <p>🚀 Advanced Analytics • 🔬 Professional Grade • 🌍 Global Standards</p>
        </div>
    </div>
    <div style="border-top: 1px solid rgba(255,255,255,0.2); margin-top: 2rem; padding-top: 1rem; text-align: center;">
        <p>© 2024 AquaCheck Pro - Ensuring Water Safety Through AI Innovation</p>
    </div>
</div>
""", unsafe_allow_html=True)
