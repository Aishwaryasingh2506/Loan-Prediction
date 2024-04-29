#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load LabelEncoders
def load_encoders():
    gender_encoder = LabelEncoder()
    married_encoder = LabelEncoder()
    education_encoder = LabelEncoder()
    self_employed_encoder = LabelEncoder()
    property_area_encoder = LabelEncoder()
    return gender_encoder, married_encoder, education_encoder, self_employed_encoder, property_area_encoder

# Preprocess the user input
def preprocess_input(data, encoders):
    gender, married, education, self_employed, property_area = data['Gender'], data['Married'], data['Education'], data['Self_Employed'], data['Property_Area']
    gender_encoded = encoders[0].transform([gender])[0]
    married_encoded = encoders[1].transform([married])[0]
    education_encoded = encoders[2].transform([education])[0]
    self_employed_encoded = encoders[3].transform([self_employed])[0]
    property_area_encoded = encoders[4].transform([property_area])[0]
    return [gender_encoded, married_encoded, education_encoded, self_employed_encoded,
            data['Applicant Income'], data['Coapplicant Income'], data['Loan Amount'], 
            data['Loan Term (in months)'], data['Credit History'], 
            property_area_encoded, data['Dependents']]

# Make prediction
def predict_loan_approval(model, data):
    prediction = model.predict([data])
    return "Approved" if prediction[0] == 1 else "Not Approved"

# Streamlit interface
def main():
    st.title('Loan Prediction App')
    
    # Load model and encoders
    model = LogisticRegression()
    encoders = load_encoders()

    # Collect user input
    gender = st.selectbox('Gender', ['Male', 'Female'])
    married = st.selectbox('Married', ['No', 'Yes'])
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['No', 'Yes'])
    applicant_income = st.number_input('Applicant Income')
    coapplicant_income = st.number_input('Coapplicant Income')
    loan_amount = st.number_input('Loan Amount')
    loan_term = st.number_input('Loan Term (in months)')
    credit_history = st.selectbox('Credit History', [1.0, 0.0])
    property_area = st.selectbox('Property Area', ['Rural', 'Semiurban', 'Urban'])
    dependents = st.number_input('Dependents')

    # Create a dictionary from user input
    user_input = {'Gender': gender, 'Married': married, 'Education': education, 
                  'Self_Employed': self_employed, 'Applicant Income': applicant_income, 
                  'Coapplicant Income': coapplicant_income, 'Loan Amount': loan_amount, 
                  'Loan Term (in months)': loan_term, 'Credit History': credit_history, 
                  'Property Area': property_area, 'Dependents': dependents}
    
    if st.button('Predict'):
        data = preprocess_input(user_input, encoders)
        result = predict_loan_approval(model, data)
        st.write(f"The loan application is {result}.")

if __name__ == '__main__':
    main()

