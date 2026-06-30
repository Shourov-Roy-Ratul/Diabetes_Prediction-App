import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

MODEL_PATH = "stacking_diabetes_model.pkl"
THRESHOLD_PATH = "stacking_best_threshold.pkl"
BACKGROUND_PATH = "shap_background.pkl"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Please upload stacking_diabetes_model.pkl.")
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error("Threshold file not found. Please upload stacking_best_threshold.pkl.")
    st.stop()

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

try:
    for name, estimator in model.named_estimators_.items():
        if hasattr(estimator, "get_booster"):
            estimator.get_booster().feature_names = None
except Exception:
    pass

if os.path.exists(BACKGROUND_PATH):
    shap_background = np.asarray(joblib.load(BACKGROUND_PATH))
else:
    shap_background = None

st.title("🩺 Diabetes Prediction App")
st.markdown(
    "Enter basic health information. The app will automatically calculate BMI "
    "and other model-required clinical features."
)

age = st.number_input("Age", min_value=1, max_value=120)
gender_text = st.radio("Gender", ["Female", "Male"])
pluse_rate = st.number_input("Pulse Rate", min_value=30, max_value=200)
systolic_bp = st.number_input("Systolic Blood Pressure (mm Hg)", min_value=50, max_value=250)
diastolic_bp = st.number_input("Diastolic Blood Pressure (mm Hg)", min_value=30, max_value=150)
glucose = st.number_input("Glucose Level (mg/dL)", min_value=40, max_value=500)
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0)

bmi = weight / ((height / 100) ** 2)
st.markdown(f"**Calculated BMI:** {bmi:.2f}")

family_diabetes = st.selectbox("Family History of Diabetes", ["No", "Yes"])
cardio_disease = st.selectbox("Cardiovascular Disease", ["No", "Yes"])
stroke = st.selectbox("Previous Stroke", ["No", "Yes"])

display_feature_names = [
    "Age", "Gender", "Pulse Rate", "Systolic BP", "Diastolic BP",
    "Glucose", "Height", "Weight", "BMI", "Family History",
    "Cardiovascular Disease", "Stroke", "Pulse Pressure",
    "BMI × Glucose", "Age × Glucose", "BP Ratio",
    "Glucose × Family History", "Age × BMI"
]

if st.button("Predict Diabetes"):
    gender = 1 if gender_text == "Male" else 0
    family_diabetes_value = 1 if family_diabetes == "Yes" else 0
    cardio_disease_value = 1 if cardio_disease == "Yes" else 0
    stroke_value = 1 if stroke == "Yes" else 0

    pulse_pressure = systolic_bp - diastolic_bp
    bmi_glucose = bmi * glucose
    age_glucose = age * glucose
    bp_ratio = systolic_bp / diastolic_bp
    glucose_family = glucose * family_diabetes_value
    age_bmi = age * bmi

    input_data = [
        age,
        gender,
        pluse_rate,
        systolic_bp,
        diastolic_bp,
        glucose,
        height,
        weight,
        bmi,
        family_diabetes_value,
        cardio_disease_value,
        stroke_value,
        pulse_pressure,
        bmi_glucose,
        age_glucose,
        bp_ratio,
        glucose_family,
        age_bmi
    ]

    input_array = np.asarray(input_data).reshape(1, -1)

    proba = model.predict_proba(input_array)[0][1]

    st.markdown(f"### Probability of Diabetes: {proba:.2%}")

    if proba >= threshold:
        st.error("⚠️ The model predicts the person is likely diabetic.")
    else:
        st.success("✅ The model predicts the person is not likely diabetic.")

    st.subheader("Calculated Clinical Features")

    calculated_df = pd.DataFrame({
        "Feature": [
            "BMI",
            "Pulse Pressure",
            "BMI × Glucose",
            "Age × Glucose",
            "BP Ratio",
            "Glucose × Family History",
            "Age × BMI"
        ],
        "Value": [
            round(bmi, 2),
            round(pulse_pressure, 2),
            round(bmi_glucose, 2),
            round(age_glucose, 2),
            round(bp_ratio, 2),
            round(glucose_family, 2),
            round(age_bmi, 2)
        ]
    })

    st.dataframe(calculated_df)

    st.subheader("🔍 SHAP Explanation")

    if shap_background is not None:
        try:
            def predict_diabetes(data):
                data = np.asarray(data)
                return model.predict_proba(data)[:, 1]

            explainer = shap.KernelExplainer(
                predict_diabetes,
                shap_background
            )

            shap_values = explainer.shap_values(
                input_array,
                nsamples=100
            )

            values = np.ravel(shap_values)

            contribution_df = pd.DataFrame({
                "Clinical Factor": display_feature_names,
                "Contribution (%)": (
                    np.abs(values) / np.sum(np.abs(values))
                ) * 100
            })

            contribution_df = contribution_df.sort_values(
                by="Contribution (%)",
                ascending=False
            )

            st.markdown("### Clinical Factor Contribution")
            st.dataframe(
                contribution_df.style.format({
                    "Contribution (%)": "{:.2f}%"
                })
            )

            st.markdown("### SHAP Bar Plot")

            fig, ax = plt.subplots(figsize=(10, 7))
            ax.barh(
                contribution_df["Clinical Factor"],
                contribution_df["Contribution (%)"]
            )
            ax.set_xlabel("Contribution (%)")
            ax.set_ylabel("Clinical Factor")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.info(
                "These percentages show how strongly each factor influenced this specific prediction. "
                "They are model explanation values, not medical diagnosis percentages."
            )

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.warning("SHAP background file not found. Please upload shap_background.pkl.")
