import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset for reference (optional, for feature names)
df = pd.read_csv('student_exam_prediction_data.csv')
feature_names = df.drop(columns=['Student_ID', 'Passed_Final_Exam']).columns

# Streamlit application
st.title('Student Exam Pass Prediction')

# Input form
age = st.number_input('Age', min_value=16, max_value=18)
gender = st.selectbox('Gender', ['M', 'F'])
previous_exam_scores = st.number_input('Previous Exam Scores', min_value=50, max_value=100)
assignment_grades = st.number_input('Assignment Grades', min_value=50, max_value=100)
attendance = st.number_input('Attendance (%)', min_value=50, max_value=100)
study_hours_per_week = st.number_input('Study Hours per Week', min_value=5, max_value=20)
participation_in_class = st.selectbox('Participation in Class', ['Low', 'Medium', 'High'])
motivation_level = st.slider('Motivation Level (1-10)', 1, 10)
stress_level = st.slider('Stress Level (1-10)', 1, 10)
socioeconomic_status = st.selectbox('Socioeconomic Status', ['Low', 'Middle', 'High'])
parental_education = st.selectbox('Parental Education', ['High School', 'Some College', 'College', 'Graduate'])
involved_in_extracurriculars = st.selectbox('Involved in Extracurriculars', ['Yes', 'No'])
sleep_hours_per_night = st.number_input('Sleep Hours per Night', min_value=5, max_value=10)
peer_average_grade = st.number_input('Peer Average Grade', min_value=50, max_value=100)

# Prepare the input data
input_data = pd.DataFrame({
    'Age': [age],
    'Previous_Exam_Scores': [previous_exam_scores],
    'Assignment_Grades': [assignment_grades],
    'Attendance (%)': [attendance],
    'Study_Hours_per_Week': [study_hours_per_week],
    'Motivation_Level (1-10)': [motivation_level],
    'Stress_Level (1-10)': [stress_level],
    'Sleep_Hours_per_Night': [sleep_hours_per_night],
    'Peer_Average_Grade': [peer_average_grade],
    'Gender_F': [1 if gender == 'F' else 0],
    'Gender_M': [1 if gender == 'M' else 0],
    'Participation_in_Class_High': [1 if participation_in_class == 'High' else 0],
    'Participation_in_Class_Low': [1 if participation_in_class == 'Low' else 0],
    'Participation_in_Class_Medium': [1 if participation_in_class == 'Medium' else 0],
    'Socioeconomic_Status_High': [1 if socioeconomic_status == 'High' else 0],
    'Socioeconomic_Status_Low': [1 if socioeconomic_status == 'Low' else 0],
    'Socioeconomic_Status_Middle': [1 if socioeconomic_status == 'Middle' else 0],
    'Parental_Education_College': [1 if parental_education == 'College' else 0],
    'Parental_Education_Graduate': [1 if parental_education == 'Graduate' else 0],
    'Parental_Education_High School': [1 if parental_education == 'High School' else 0],
    'Parental_Education_Some College': [1 if parental_education == 'Some College' else 0],
    'Involved_in_Extracurriculars_No': [1 if involved_in_extracurriculars == 'No' else 0],
    'Involved_in_Extracurriculars_Yes': [1 if involved_in_extracurriculars == 'Yes' else 0]
})

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)
    result = 'Pass' if prediction[0] == 1 else 'Fail'
    st.write(f'The student is predicted to: {result}')

# To run this Streamlit app, save it as `app.py` and use the following command:
# streamlit run app.py
