import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("RandomForestModel.pkl", 'rb'))

st.title("Diabetes Detector")

Pregnancies = int(st.number_input(label="Enter the number of Pregnancies if you are male type 0"))

Glucose = int(st.number_input(label="Enter your Glucose level"))

BloodPressure = float(st.number_input(label="Enter your BloodPressure"))

SkinThickness = float(st.number_input(label="Enter your SkinThickness"))

Insulin = float(st.number_input(label="Enter your Insulin level"))

BMI = float(st.number_input(label="Enter your BMI"))

DiabetesPedigreeFunction = float(st.number_input(label="Enter your Diabetes Pedigree Function "))

age = int(st.number_input(label="Enter your age"))

data_array = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, age]])

predict = st.button("Predict")

if (predict):
    
    y_hat = model.predict(data_array)
    
    if y_hat == 1:
        y_hat = "You maybe get diabetes"
    else:
        y_hat = "You will not get diabetes"
    
    st.success(y_hat)
    
# streamlit run app.py