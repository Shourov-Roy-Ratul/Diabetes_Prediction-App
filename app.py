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
    st.error("Model file not found. Please upload stacking_diabetes_model.pkl to GitHub.")
    st.stop()

if not os.path.exists(THRESHOLD_PATH):
    st.error("Threshold file not found. Please upload stacking_best_threshold.pkl to GitHub.")
    st.stop()

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)

# Fix XGBoost feature-name validation issue inside StackingClassifier
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
st.markdown("Enter the following health information to predict diabetes risk.")

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

family_diabetes = st.selectbox("Family History of Diabetes", [0, 1])
cardio_disease = st.selectbox("Cardiovascular Disease", [0, 1])
stroke = st.selectbox("Previous Stroke", [0, 1])

display_feature_names = [
    "Age",
    "Gender",
    "Pulse Rate",
    "Systolic BP",
    "Diastolic BP",
    "Glucose",
    "Height",
    "Weight",
    "BMI",
    "Family History",
    "Cardiovascular Disease",
    "Stroke"
]

if st.button("Predict Diabetes"):
    gender = 1 if gender_text == "Male" else 0

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
        family_diabetes,
        cardio_disease,
        stroke
    ]

    input_array = np.asarray(input_data).reshape(1, -1)

    proba = model.predict_proba(input_array)[0][1]

    st.markdown(f"### Probability of Diabetes: {proba:.2%}")

    if proba >= threshold:
        st.error("⚠️ The model predicts the person is likely diabetic.")
    else:
        st.success("✅ The model predicts the person is not likely diabetic.")

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
                "SHAP Value": values
            })

            contribution_df["Contribution (%)"] = (
                contribution_df["SHAP Value"].abs()
                / contribution_df["SHAP Value"].abs().sum()
                * 100
            )

            contribution_df = contribution_df.sort_values(
                by="Contribution (%)",
                ascending=False
            )

            st.markdown("### Clinical Factor Contribution")

            st.dataframe(
                contribution_df[["Clinical Factor", "Contribution (%)"]]
                .style.format({"Contribution (%)": "{:.2f}%"})
            )

            st.markdown("### SHAP Bar Plot")

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(
                contribution_df["Clinical Factor"],
                contribution_df["Contribution (%)"]
            )
            ax.set_xlabel("Contribution (%)")
            ax.set_ylabel("Clinical Factor")
            ax.invert_yaxis()
            st.pyplot(fig)

            st.info(
                "These percentages show how strongly each clinical factor influenced this specific prediction. "
                "They are model explanation values, not medical diagnosis percentages."
            )

        except Exception as e:
            st.warning(f"SHAP explanation unavailable: {e}")
    else:
        st.warning(
            "SHAP background file not found. Please upload shap_background.pkl to enable SHAP explanation."
        )
