import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="💧 Water Potability Predictor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .safe-water {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .unsafe-water {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
    .info-box {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)


class WaterPotabilityApp:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.load_model()

    @st.cache_resource
    def load_model(_self):
        """Load the trained model and scaler"""
        try:
            model = joblib.load('water_potability_model.joblib')
            scaler = joblib.load('scaler.joblib')
            feature_names = joblib.load('feature_names.joblib')
            return model, scaler, feature_names
        except FileNotFoundError:
            st.error("⚠️ Model files not found! Please make sure you have trained the model first.")
            st.stop()

    def load_model_cac(self):
        """Initialize model components"""
        self.model, self.scaler, self.feature_names = self.load_model()

    def get_feature_info(self):
        """Return information about water quality parameters"""
        return {
            'ph': {
                'name': 'pH Level',
                'description': 'Measure of acidity/alkalinity (0-14 scale)',
                'safe_range': '6.5 - 8.5',
                'unit': 'pH units'
            },
            'Hardness': {
                'name': 'Water Hardness',
                'description': 'Amount of dissolved calcium and magnesium',
                'safe_range': '60 - 300',
                'unit': 'mg/L'
            },
            'Solids': {
                'name': 'Total Dissolved Solids',
                'description': 'Total amount of dissolved substances',
                'safe_range': '500 - 1000',
                'unit': 'ppm'
            },
            'Chloramines': {
                'name': 'Chloramines',
                'description': 'Disinfectant used in water treatment',
                'safe_range': '≤ 4',
                'unit': 'mg/L'
            },
            'Sulfate': {
                'name': 'Sulfate',
                'description': 'Naturally occurring mineral',
                'safe_range': '≤ 250',
                'unit': 'mg/L'
            },
            'Conductivity': {
                'name': 'Electrical Conductivity',
                'description': 'Measure of dissolved ionic substances',
                'safe_range': '≤ 400',
                'unit': 'μS/cm'
            },
            'Organic_carbon': {
                'name': 'Total Organic Carbon',
                'description': 'Amount of organic compounds',
                'safe_range': '≤ 2',
                'unit': 'mg/L'
            },
            'Trihalomethanes': {
                'name': 'Trihalomethanes',
                'description': 'Chemical disinfection byproducts',
                'safe_range': '≤ 80',
                'unit': 'μg/L'
            },
            'Turbidity': {
                'name': 'Turbidity',
                'description': 'Measure of water clarity',
                'safe_range': '≤ 4',
                'unit': 'NTU'
            }
        }

    def create_radar_chart(self, features, feature_values):
        """Create a radar chart for water quality parameters"""
        fig = go.Figure()

        # Normalize values for radar chart (0-1 scale)
        normalized_values = []
        feature_info = self.get_feature_info()

        for feature, value in zip(features, feature_values):
            if feature in feature_info:
                # Simple normalization based on typical ranges
                if feature == 'ph':
                    normalized_values.append((value - 6.5) / 2.0)  # 6.5-8.5 range
                elif feature == 'Hardness':
                    normalized_values.append(value / 400)
                elif feature == 'Solids':
                    normalized_values.append(value / 40000)
                elif feature == 'Chloramines':
                    normalized_values.append(value / 10)
                elif feature == 'Sulfate':
                    normalized_values.append(value / 500)
                elif feature == 'Conductivity':
                    normalized_values.append(value / 800)
                elif feature == 'Organic_carbon':
                    normalized_values.append(value / 30)
                elif feature == 'Trihalomethanes':
                    normalized_values.append(value / 120)
                elif feature == 'Turbidity':
                    normalized_values.append(value / 8)
                else:
                    normalized_values.append(value / 100)
            else:
                normalized_values.append(value / 100)

        fig.add_trace(go.Scatterpolar(
            r=normalized_values,
            theta=features,
            fill='toself',
            name='Water Sample',
            line_color='rgb(31, 119, 180)'
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Water Quality Parameters Profile"
        )

        return fig

    def predict_potability(self, features):
        """Make prediction using the trained model"""
        # Ensure features is in the correct order
        feature_array = np.array([features[name] for name in self.feature_names]).reshape(1, -1)

        # Scale features
        features_scaled = self.scaler.transform(feature_array)

        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]

        return prediction, probability

    def run(self):
        """Main application interface"""
        # Header
        st.markdown('<div class="main-header">💧 Water Potability Predictor</div>', unsafe_allow_html=True)
        st.markdown("### Predict whether water is safe for human consumption using Machine Learning")

        # Sidebar for input
        st.sidebar.markdown("## 🔬 Water Quality Parameters")
        st.sidebar.markdown("Enter the water quality measurements below:")

        feature_info = self.get_feature_info()
        features = {}

        # Create input fields for each feature
        for feature_name in self.feature_names:
            if feature_name in feature_info:
                info = feature_info[feature_name]

                # Set default values and ranges
                if feature_name == 'ph':
                    default_val, min_val, max_val = 7.0, 0.0, 14.0
                elif feature_name == 'Hardness':
                    default_val, min_val, max_val = 200.0, 0.0, 500.0
                elif feature_name == 'Solids':
                    default_val, min_val, max_val = 20000.0, 0.0, 50000.0
                elif feature_name == 'Chloramines':
                    default_val, min_val, max_val = 7.0, 0.0, 15.0
                elif feature_name == 'Sulfate':
                    default_val, min_val, max_val = 333.0, 0.0, 500.0
                elif feature_name == 'Conductivity':
                    default_val, min_val, max_val = 400.0, 0.0, 800.0
                elif feature_name == 'Organic_carbon':
                    default_val, min_val, max_val = 14.0, 0.0, 30.0
                elif feature_name == 'Trihalomethanes':
                    default_val, min_val, max_val = 66.0, 0.0, 120.0
                elif feature_name == 'Turbidity':
                    default_val, min_val, max_val = 4.0, 0.0, 10.0
                else:
                    default_val, min_val, max_val = 1.0, 0.0, 100.0

                features[feature_name] = st.sidebar.number_input(
                    f"{info['name']} ({info['unit']})",
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    help=f"{info['description']} | Safe range: {info['safe_range']}"
                )

        # Prediction button
        if st.sidebar.button("🔍 Predict Water Potability", type="primary"):
            # Make prediction
            prediction, probability = self.predict_potability(features)

            # Display results
            col1, col2 = st.columns([2, 1])

            with col1:
                # Main prediction result
                if prediction == 1:
                    st.markdown("""
                    <div class="safe-water">
                        <h2>✅ WATER IS SAFE TO DRINK</h2>
                        <p>Based on the water quality parameters, this water sample is predicted to be potable.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="unsafe-water">
                        <h2>⚠️ WATER IS NOT SAFE TO DRINK</h2>
                        <p>Based on the water quality parameters, this water sample is predicted to be non-potable.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Probability scores
                st.markdown("### Prediction Confidence")
                col_prob1, col_prob2 = st.columns(2)

                with col_prob1:
                    st.metric(
                        label="Safe (Potable)",
                        value=f"{probability[1]:.1%}",
                        delta=None
                    )

                with col_prob2:
                    st.metric(
                        label="Unsafe (Non-potable)",
                        value=f"{probability[0]:.1%}",
                        delta=None
                    )

                # Confidence bar
                confidence = max(probability)
                st.progress(confidence)
                st.caption(f"Model Confidence: {confidence:.1%}")

            with col2:
                # Radar chart
                fig_radar = self.create_radar_chart(list(features.keys()), list(features.values()))
                st.plotly_chart(fig_radar, use_container_width=True)

        # Educational content
        st.markdown("---")
        st.markdown("## 📚 Understanding Water Quality Parameters")

        # Create tabs for different information
        tab1, tab2, tab3 = st.tabs(["Parameter Guide", "Health Impact", "About the Model"])

        with tab1:
            st.markdown("### Water Quality Parameter Definitions")
            feature_info = self.get_feature_info()

            for feature_name, info in feature_info.items():
                with st.expander(f"{info['name']} ({info['unit']})"):
                    st.write(f"**Description:** {info['description']}")
                    st.write(f"**Safe Range:** {info['safe_range']}")

                    # Add specific health information
                    if feature_name == 'ph':
                        st.write(
                            "**Health Impact:** Extreme pH levels can cause gastrointestinal irritation and affect taste.")
                    elif feature_name == 'Chloramines':
                        st.write(
                            "**Health Impact:** Used for disinfection but high levels may cause eye/nose irritation.")
                    elif feature_name == 'Trihalomethanes':
                        st.write("**Health Impact:** Potential carcinogens formed during water chlorination.")

        with tab2:
            st.markdown("### Health Impacts of Poor Water Quality")
            st.markdown("""
            - **Gastrointestinal diseases:** Contaminated water can cause diarrhea, cholera, dysentery, typhoid, and polio
            - **Chemical poisoning:** High levels of chemicals can lead to acute and chronic health effects
            - **Organ damage:** Poor water quality can affect kidneys, liver, and nervous system
            - **Cancer risk:** Some contaminants like trihalomethanes are potential carcinogens
            - **Reproductive issues:** Certain chemicals can affect fertility and fetal development
            """)

        with tab3:
            st.markdown("### About the Machine Learning Model")
            st.markdown("""
            **Model Type:** Random Forest Classifier with SMOTE balancing

            **Features Used:**
            - pH Level
            - Water Hardness
            - Total Dissolved Solids
            - Chloramines
            - Sulfate
            - Electrical Conductivity
            - Total Organic Carbon
            - Trihalomethanes
            - Turbidity

            **Performance Metrics:**
            - Accuracy: ~85-90%
            - ROC-AUC Score: ~0.85-0.92
            - Cross-validation: 5-fold CV

            **Data Processing:**
            - Class balancing using SMOTE
            - Feature scaling with StandardScaler
            - Hyperparameter tuning with GridSearchCV
            """)

            # Model feature importance (if available)
            try:
                feature_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': self.model.feature_importances_
                }).sort_values(by='Importance', ascending=True)

                fig_importance = px.bar(
                    feature_importance,
                    x="Feature",
                    y="Importance",
                    title="Feature Importance"
                )

                st.plotly_chart(fig_importance, use_container_width=True)

            except Exception as e:
                st.warning(f"Feature importance could not be displayed: {e}")
