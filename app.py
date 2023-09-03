import streamlit as st
import pandas as pd
import numpy as np
import pickle



st.set_page_config(page_title="Diabetes Prediction")

st.title('Diabetes Checkup')

model = pickle.load(open('model.pkl','rb'))

def user_report():
  pregnancies = st.number_input('Pregnancies', 0,17, 0 )
  glucose =  st.number_input('Glucose', 0,200, 180 )
  bp =  st.number_input('Blood Pressure', 0,122, 90 )
  skinthickness =  st.number_input('Skin Thickness', 0,100, 26 )
  insulin =  st.number_input('Insulin', 0,846, 90 )
  bmi =  st.number_input('BMI', 0,67, 37 )
  dpf =  st.number_input('Diabetes Pedigree Function', 0.0,2.4, 0.314 )
  age =  st.number_input('Age', 21,88, 35 )

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
  else:
    st.title("You may not have a diabetes")

  st.markdown("A PROJECT BY - M.MANOJ BHASKAR")








