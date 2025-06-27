
import streamlit as st
import numpy as np
import joblib

model = joblib.load("rf_diabetes_model_all_features.pkl")
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

if st.button("Predict Diabetes"):
    gender = 1 if gender_text == "Male" else 0

    input_data = (
        age, gender, pluse_rate, systolic_bp, diastolic_bp, glucose,
        height, weight, bmi, family_diabetes, cardio_disease, stroke
    )

    input_array = np.asarray(input_data).reshape(1, -1)
    proba = model.predict_proba(input_array)[0][1]

    threshold = 0.6

    st.markdown(f"**Probability of Diabetes:** {proba:.2%}")
    if proba >= threshold:
        st.error("⚠️ The model predicts the person is **likely diabetic**.")
    else:
        st.success("✅ The model predicts the person is **not likely diabetic**.")
