import streamlit as st
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Student Exam Performance Indicator", layout="centered")

st.title("📊 Student Exam Performance Indicator")
st.write("Fill in the details below to predict your **Math Score**.")

# ✅ Mapping dictionaries to match training categories exactly
gender_map = {"Male": "male", "Female": "female"}
ethnicity_map = {
    "Group A": "group A",
    "Group B": "group B",
    "Group C": "group C",
    "Group D": "group D",
    "Group E": "group E"
}
parent_edu_map = {
    "associate's degree": "associate's degree",
    "bachelor's degree": "bachelor's degree",
    "high school": "high school",
    "master's degree": "master's degree",
    "some college": "some college",
    "some high school": "some high school"
}
lunch_map = {"free/reduced": "free/reduced", "standard": "standard"}
course_map = {"none": "none", "completed": "completed"}

# Input form
with st.form("student_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Race or Ethnicity", ["Group A", "Group B", "Group C", "Group D", "Group E"])
    parental_level_of_education = st.selectbox(
        "Parental Level of Education",
        ["associate's degree", "bachelor's degree", "high school",
        "master's degree", "some college", "some high school"]
    )
    lunch = st.selectbox("Lunch Type", ["free/reduced", "standard"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["none", "completed"])
    
    reading_score = st.number_input("Reading Score (0-100)", min_value=0, max_value=100, step=1)
    writing_score = st.number_input("Writing Score (0-100)", min_value=0, max_value=100, step=1)

    submitted = st.form_submit_button("🔮 Predict Math Score")

# ----------------- Prediction -----------------
if submitted:
    try:
        # ✅ Map inputs to match training categories
        data = CustomData(
            gender=gender_map[gender],
            race_ethnicity=ethnicity_map[ethnicity],
            parental_level_of_education=parent_edu_map[parental_level_of_education],
            lunch=lunch_map[lunch],
            test_preparation_course=course_map[test_preparation_course],
            reading_score=reading_score,
            writing_score=writing_score
        )

        pred_df = data.get_data_as_data_frame()
        st.write("### 📑 Input Data Preview")
        st.dataframe(pred_df)

        # Predict
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        st.success(f"✅ Predicted Math Score: **{results[0]:.2f}**")

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
