import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ── Load Saved Files ─────────────────────────────────────────────────
# We load once and cache — so the app does not reload on every interaction
@st.cache_resource
def load_model():
    model        = joblib.load('diabetes_model.pkl')
    scaler       = joblib.load('diabetes_scaler.pkl')
    feature_names= joblib.load('feature_names.pkl')
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# ── Header ───────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.markdown("#### Explainable Clinical Decision Support Tool")
st.markdown("""
This tool predicts diabetes risk from clinical measurements using an XGBoost 
model trained on the PIMA Indians Diabetes Dataset (ROC-AUC: 0.95).  
Enter patient measurements in the sidebar and click **Predict** to receive 
a risk assessment with a full explanation of contributing factors.
""")
st.divider()

# ── Sidebar — Patient Input ───────────────────────────────────────────
st.sidebar.title("🔬 Patient Measurements")
st.sidebar.markdown("Adjust the sliders to match the patient's clinical values.")

pregnancies = st.sidebar.slider(
    "Pregnancies",
    min_value=0, max_value=20, value=3,
    help="Number of times pregnant"
)
glucose = st.sidebar.slider(
    "Glucose (mg/dL)",
    min_value=50, max_value=250, value=120,
    help="Plasma glucose concentration (2-hour oral glucose tolerance test)"
)
blood_pressure = st.sidebar.slider(
    "Blood Pressure (mm Hg)",
    min_value=30, max_value=130, value=70,
    help="Diastolic blood pressure"
)
skin_thickness = st.sidebar.slider(
    "Skin Thickness (mm)",
    min_value=5, max_value=100, value=28,
    help="Tricep skin fold thickness"
)
insulin = st.sidebar.slider(
    "Insulin (µU/mL)",
    min_value=15, max_value=850, value=120,
    help="2-hour serum insulin"
)
bmi = st.sidebar.slider(
    "BMI",
    min_value=10.0, max_value=70.0, value=32.0, step=0.1,
    help="Body Mass Index"
)
dpf = st.sidebar.slider(
    "Diabetes Pedigree Function",
    min_value=0.05, max_value=2.50, value=0.47, step=0.01,
    help="Genetic risk score based on family history"
)
age = st.sidebar.slider(
    "Age (years)",
    min_value=18, max_value=90, value=35,
    help="Patient age in years"
)

predict_button = st.sidebar.button("🔍 Predict", use_container_width=True)

# ── Prediction Logic ──────────────────────────────────────────────────
if predict_button:

    # Assemble input into DataFrame
    input_data = pd.DataFrame([[
        pregnancies, glucose, blood_pressure, skin_thickness,
        insulin, bmi, dpf, age
    ]], columns=feature_names)

    # Scale using the saved scaler
    input_scaled = scaler.transform(input_data)

    # Predict
    probability  = model.predict_proba(input_scaled)[0][1]
    prediction   = model.predict(input_scaled)[0]

    # ── Result Display ────────────────────────────────────────────────
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Prediction Result")

        if prediction == 1:
            st.error(f"### ⚠️ High Diabetes Risk Detected")
            st.markdown(f"**Probability of Diabetes: {probability:.1%}**")
            st.markdown("""
            > This patient shows clinical indicators consistent with diabetes.  
            > A confirmatory HbA1c or fasting glucose test is recommended.
            """)
        else:
            st.success(f"### ✅ Low Diabetes Risk")
            st.markdown(f"**Probability of Diabetes: {probability:.1%}**")
            st.markdown("""
            > This patient's clinical measurements do not indicate elevated 
            > diabetes risk at this time. Routine monitoring is advised.
            """)

        # Risk probability gauge
        st.markdown("#### Risk Probability")
        st.progress(float(probability))
        st.caption(f"{probability:.1%} probability of diabetes")

        # Patient summary table
        st.markdown("#### Patient Measurements Summary")
        summary = pd.DataFrame({
            'Feature': feature_names,
            'Value': [pregnancies, glucose, blood_pressure,
                      skin_thickness, insulin, bmi, dpf, age]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("🔍 Why This Prediction? (SHAP Explanation)")
        st.markdown("""
        The chart below shows which features pushed the prediction 
        **toward diabetes (red)** or **away from diabetes (blue)**.
        """)

        # Calculate SHAP values for this patient
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # SHAP waterfall plot
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_data.iloc[0],
                feature_names=feature_names
            ),
            show=False
        )
        plt.title("Feature Contributions to Prediction", 
                  fontweight='bold', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Feature importance bar chart
        st.markdown("#### Feature Impact Ranking")
        shap_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP Value': np.abs(shap_values[0])
        }).sort_values('SHAP Value', ascending=False)

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        colors = ['#e74c3c' if v > 0 else '#3498db' 
                  for v in shap_values[0][
                      np.argsort(np.abs(shap_values[0]))[::-1]
                  ]]
        ax2.barh(shap_df['Feature'], shap_df['SHAP Value'], 
                 color=colors, edgecolor='white')
        ax2.set_xlabel('Absolute SHAP Value (Impact on Prediction)')
        ax2.set_title('Feature Importance for This Patient', fontweight='bold')
        ax2.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.divider()

    # ── Clinical Interpretation ───────────────────────────────────────
    st.subheader("📖 Clinical Interpretation")
    col3, col4, col5 = st.columns(3)

    with col3:
        glucose_status = "⚠️ Elevated" if glucose > 125 else "✅ Normal"
        st.metric("Glucose Status", glucose_status, 
                  f"{glucose} mg/dL (Normal: 70-125)")

    with col4:
        bmi_status = "⚠️ Obese" if bmi >= 30 else ("⚠️ Overweight" if bmi >= 25 else "✅ Normal")
        st.metric("BMI Status", bmi_status, 
                  f"{bmi} (Normal: 18.5-24.9)")

    with col5:
        bp_status = "⚠️ Elevated" if blood_pressure > 80 else "✅ Normal"
        st.metric("Blood Pressure", bp_status, 
                  f"{blood_pressure} mm Hg (Normal: <80)")

    st.divider()
    st.caption("""
    ⚠️ Disclaimer: This tool is for educational and research purposes only. 
    It is not a substitute for professional medical diagnosis or clinical judgment. 
    All predictions should be reviewed by a qualified healthcare professional.
    """)

else:
    # Default state before prediction
    st.info("👈 Enter patient measurements in the sidebar and click **Predict** to begin.")

    st.markdown("### About This Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC", "0.95")
    col2.metric("Diabetic Recall", "85%")
    col3.metric("Training Patients", "614")
    col4.metric("Test Patients", "154")

    st.markdown("### How It Works")
    st.markdown("""
    1. **Enter** the patient's clinical measurements using the sliders on the left
    2. **Click Predict** to run the XGBoost model
    3. **Review** the risk probability and SHAP explanation
    4. **Interpret** which clinical features contributed most to the prediction
    """)

    st.markdown("### Model Pipeline")
    st.markdown("""
    - **Data Cleaning**: Conditional median imputation of biologically impossible zero values
    - **Imbalance Handling**: SMOTE applied exclusively to training data
    - **Model**: XGBoost classifier (200 trees, max depth 4, learning rate 0.1)
    - **Explainability**: SHAP TreeExplainer for individual prediction interpretation
    - **Evaluation**: ROC-AUC 0.95, Diabetic Recall 85%, Cross-Validation Mean AUC 0.97
    """)