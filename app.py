import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Page configuration
st.set_page_config(page_title="Diabetes Prediction Checkup", page_icon="🩺", layout="centered")

# Add a header and description
st.markdown(
    """
    <style>
    .title {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #4CAF50;
        margin-bottom: 20px;
    }
    .description {
        font-size: 1.2em;
        text-align: justify;
        margin-bottom: 30px;
    }
    .footer {
        font-size: 1em;
        text-align: center;
        margin-top: 50px;
        color: #666;
    }
    .result {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="title">Diabetes Prediction Checkup</div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="description">
    This tool predicts whether or not someone may have diabetes using a Support Vector Classifier (SVC).
    The model was trained with an accuracy of 80%. 
    <br><br>
    <b>Disclaimer:</b> This is not a substitute for medical advice. Please consult a healthcare professional for concerns regarding your health.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

# User input form
def user_report():
    st.sidebar.header("Enter Your Details:")
    age = st.sidebar.number_input('Age', 21, 88, 35)
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 0)
    glucose = st.sidebar.number_input('Glucose', 0, 200, 180)
    bp = st.sidebar.slider('Blood Pressure', 0, 122, 90)
    skinthickness = st.sidebar.number_input('Skin Thickness', 0, 100, 26)
    insulin = st.sidebar.slider('Insulin', 0, 846, 90)
    bmi = st.sidebar.number_input('BMI', 0, 67, 37)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.0, 2.4, 0.314)
    
    # Combine inputs into a dataframe
    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Get input data
input_df = user_report()

# Load dataset for input preprocessing
dataset = pd.read_csv('diabetes.csv')
dataset = dataset.drop(columns=['Outcome'])
df = pd.concat([input_df, dataset], axis=0)
df = df[:1]  # Use only the first row

# Predict and display results
if st.button('Predict 🚀'):
    prediction = model.predict(df)
    if prediction[0] == 1:
        st.markdown('<div class="result" style="color: red;">You may have diabetes.</div>', unsafe_allow_html=True)
        st.write("⚠️ It's important to consult a healthcare professional for a comprehensive evaluation and guidance on managing your health.")
    else:
        st.markdown('<div class="result" style="color: green;">You may not have diabetes.</div>', unsafe_allow_html=True)
        st.write("✅ Maintaining a healthy lifestyle is key to preventing diabetes. Continue to prioritize a balanced diet and regular exercise.")
    
    # Footer
    st.markdown('<div class="footer">A PROJECT BY - M. MANOJ BHASKAR</div>', unsafe_allow_html=True)
