import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# Load the cleaned data
df = pd.read_csv('../data/transformed_data.csv')

# Load the trained model
model = joblib.load('../models/survival_predictor.pkl')

st.markdown(
    """
    <div style="font-family: 'Helvetica Neue', Arial, sans-serif; color: white; padding: 15px; text-align: center;
    background: linear-gradient(to right, #3498db, #2980b9); border-radius: 8px; margin-bottom: 20px;">
        <h2>PREDICTING LUNG CANCER SURVIVAL</h2>
    </div>
    """, unsafe_allow_html=True
)

st.sidebar.header("Patient Demographics")
age = st.sidebar.slider('Age', 0, 100, 50)
gender = st.sidebar.selectbox('Gender', df['gender'].unique())
country = st.sidebar.selectbox('Country', df['country'].unique())

st.sidebar.header("Clinical Information")
cancer_stage = st.sidebar.selectbox('Stage', df['cancer_stage'].unique())
comorbidities_count = st.sidebar.slider('Comorbidities', 0, 4, 0)

st.sidebar.header("Health & Lifestyle Factors")
smoking_status = st.sidebar.selectbox('Smoking', df['smoking_status'].unique())
bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
cholesterol_level = st.sidebar.slider('Cholesterol', 100, 300, 200)

st.sidebar.header("Treatment Details")
treatment_type = st.sidebar.selectbox('Treatment', df['treatment_type'].unique())
treatment_duration = st.sidebar.slider('Duration (days)', 0, 365, 30)

if st.button('Predict Survival'):
    current_date = datetime.now()
    sample_input = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'country': [country],
        'cancer_stage': [cancer_stage],
        'smoking_status': [smoking_status],
        'bmi': [bmi],
        'cholesterol_level': [cholesterol_level],
        'treatment_type': [treatment_type],
        'treatment_duration': [treatment_duration],
        'comorbidities_count': [comorbidities_count],
        'id': [1],
        'diagnosis_year': [current_date.year],
        'diagnosis_month': [current_date.month],
        'diagnosis_quarter': [(current_date.month - 1) // 3 + 1],
        'hypertension': [0],
        'asthma': [0],
        'cirrhosis': [0],
        'other_cancer': [0],
        'family_history': [0]
    })

    sample_input['age_group'] = pd.cut(sample_input['age'],
                                       bins=[0, 18, 30, 45, 60, 75, 100],
                                       labels=['<18', '18-29', '30-44', '45-59', '60-74', '75+'])

    sample_input['bmi_category'] = pd.cut(sample_input['bmi'],
                                          bins=[0, 18.5, 24.9, 29.9, 100],
                                          labels=['underweight', 'normal', 'overweight', 'obese'])

    sample_input['cholesterol_category'] = pd.cut(sample_input['cholesterol_level'],
                                                  bins=[0, 200, 239, 1000],
                                                  labels=['Desirable', 'Borderline high', 'High'])

    prediction = model.predict_proba(sample_input)[0]

    st.markdown(
        f"""
        <div style="padding: 15px; background-color: #e8f6f3; border-left: 5px solid #2ecc71;
        border-radius: 4px; margin: 10px 0; font-family: Arial, sans-serif;">
            <h3 style="color: #27ae60; margin: 0 0 10px 0;">Prediction Result</h3>
            <p style="font-size: 18px; margin: 0;">
                Survival Probability: <strong>{prediction[1]:.1%}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True
    )
