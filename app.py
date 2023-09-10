import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Diabetes Prediction")

st.title('Diabetes Prediction Checkup')

st.write("This model attempts to predict whether or not someone has Diabetes using the SVC CLASSIFIER.\n\nThe model was trained with 80% percent accuracy\n\nPlease note that this should not be used as a medical diagnosis, rather just a tool to help. Please consult your local medical professional if you have any health concerns.",width=500)

model = pickle.load(open('model.pkl','rb'))

def user_report():
  age =  st.number_input('Age', 21,88, 35 )

  pregnancies = st.slider('Pregnancies', 0,17, 0 )

  glucose =  st.number_input('Glucose', 0,200, 180 )

  bp =  st.slider('Blood Pressure', 0,122, 90 )

  skinthickness =  st.number_input('Skin Thickness', 0,100, 26 )

  insulin =  st.slider('Insulin', 0,846, 90 )

  bmi =  st.number_input('BMI', 0,67, 37 )

  dpf =  st.slider('Diabetes Pedigree Function', 0.0,2.4, 0.314 )
 

  user_report_data = {

      'Pregnancies':pregnancies,

      'Glucose':glucose,

      'BloodPressure':bp,

      'SkinThickness':skinthickness,

      'Insulin':insulin,

      'BMI':bmi,

      'DiabetesPedigreeFunction':dpf,

      'Age':age
  }
  report_data = pd.DataFrame(user_report_data, index=[0])

  return report_data

input_df = user_report()

dataset = pd.read_csv('diabetes.csv')

dataset = dataset.drop(columns=['Outcome'])

df = pd.concat([input_df,dataset],axis = 0)

df = df[:1]

if st.button('predict'):

  prediction = model.predict(df)

  if prediction[0] == 1:
    st.title("You may have diabetes.")
    st.write("It's important to consult a healthcare professional for a comprehensive evaluation and guidance on managing your health.")

  else:
    st.title("You may not have diabetes.")
    st.write("Maintaining a healthy lifestyle is key to preventing diabetes. Continue to prioritize a balanced diet and regular exercise.")



st.markdown("A PROJECT BY - M.MANOJ BHASKAR")








