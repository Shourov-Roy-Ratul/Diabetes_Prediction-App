import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

model = joblib.load("stacking_diabetes_model.pkl")
threshold = joblib.load("stacking_best_threshold.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")

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
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
family_diabetes = st.selectbox("Family History of Diabetes", [0, 1])
cardio_disease = st.selectbox("Cardiovascular Disease", [0, 1])
stroke = st.selectbox("Previous Stroke", [0, 1])

feature_names = [
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

model_feature_names = [
    "age",
    "gender",
    "pluse_rate",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "height",
    "weight",
    "bmi",
    "family_diabetes",
    "cardiovascular_disease",
    "stroke"
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

    input_df = pd.DataFrame([input_data], columns=model_feature_names)

    proba = model.predict_proba(input_df)[0][1]

    st.markdown(f"### Probability of Diabetes: {proba:.2%}")

    if proba >= threshold:
        st.error("⚠️ The model predicts the person is likely diabetic.")
    else:
        st.success("✅ The model predicts the person is not likely diabetic.")

    st.subheader("🔍 Clinical Factor Contribution (SHAP)")

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        if isinstance(shap_values, list):
            values = np.ravel(shap_values[1][0])
            base_value = explainer.expected_value[1]
        else:
            shap_values_array = np.array(shap_values)

            if shap_values_array.ndim == 3:
                values = np.ravel(shap_values_array[0, :, 1])
                base_value = explainer.expected_value[1]
            else:
                values = np.ravel(shap_values_array[0])
                base_value = explainer.expected_value

        contribution_df = pd.DataFrame({
            "Clinical Factor": feature_names,
            "Contribution (%)": (np.abs(values) / np.sum(np.abs(values))) * 100
        })

        contribution_df = contribution_df.sort_values(
            by="Contribution (%)",
            ascending=False
        )

        st.dataframe(
            contribution_df.style.format({
                "Contribution (%)": "{:.2f}%"
            })
        )

        st.subheader("📊 SHAP Waterfall Explanation")

        explanation = shap.Explanation(
            values=values,
            base_values=base_value,
            data=input_df.iloc[0].values,
            feature_names=feature_names
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")
