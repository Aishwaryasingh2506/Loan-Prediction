import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("Loan_Data.csv")

df.loc[df.isnull().any(axis=1)]

df.drop('Loan_ID', axis=1, inplace=True)

na_columns = ['Gender', "Married", 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for na_column in na_columns:
    med = df[na_column].mode()[0]
    df[na_column] = df[na_column].fillna(med) 

df.Gender = (df.Gender == 'Male').astype(int)
df.Married = (df.Married == 'Yes').astype(int)
df.Self_Employed = (df.Self_Employed == 'Yes').astype(int)
df.Education = (df.Education == 'Graduate').astype(int)
df.Loan_Status = (df.Loan_Status == 'Y').astype(int)
df.Dependents = df.Dependents.replace(to_replace="3+", value="3").astype(int)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.median())

dummies = pd.get_dummies(df)

X = dummies.drop(['Loan_Status'], axis=1)
y = dummies.Loan_Status

print(X.columns)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Load the pre-trained model
log = LogisticRegression()
log.fit(X_train, y_train)

# Load label encoders
label_encoders = {
    "Gender": LabelEncoder().fit(['Female', 'Male']),
    "Married": LabelEncoder().fit(['No', 'Yes']),
    "Education": LabelEncoder().fit(['Graduate', 'Not Graduate']),
    "Self_Employed": LabelEncoder().fit(['No', 'Yes']),
    "Property_Area": LabelEncoder().fit(['Rural', 'Semiurban', 'Urban'])
}

# Define the UI elements
st.title("Loan Approval Prediction")

gender = st.selectbox("Gender", ['Female', 'Male'])
married = st.selectbox("Married", ['No', 'Yes'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['No', 'Yes'])
applicant_income = st.number_input("Applicant Income", value=0)
coapplicant_income = st.number_input("Coapplicant Income", value=0)
loan_amount = st.number_input("Loan Amount", value=0)
loan_term = st.number_input("Loan Term (in months)", value=0)
credit_history = st.number_input("Credit History (1.0 or 0.0)", value=0.0)
property_area = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])
dependents = st.number_input("Number of Dependents", value=0)

Property_Area_Rural = 0
Property_Area_Semiurban = 0
Property_Area_Urban = 0

if property_area == "Rural":
    Property_Area_Rural = 1
elif property_area == "Semiurban":
    Property_Area_Semiurban = 1
elif property_area == "Urban":
    Property_Area_Urban = 1

# Encode categorical variables
gender_encoded = label_encoders["Gender"].transform([gender])[0]
married_encoded = label_encoders["Married"].transform([married])[0]
education_encoded = label_encoders["Education"].transform([education])[0]
self_employed_encoded = label_encoders["Self_Employed"].transform([self_employed])[0]
property_area_encoded = label_encoders["Property_Area"].transform([property_area])[0]

# Make prediction
if st.button("Predict"):
    prediction = log.predict([[gender_encoded, married_encoded, education_encoded, self_employed_encoded,
             applicant_income, coapplicant_income, loan_amount, loan_term, credit_history, Property_Area_Rural, Property_Area_Semiurban, Property_Area_Urban, dependents]])
    result = "Approved" if prediction[0] == 1 else "Not Approved"
    st.write(f"The loan application is {result}.")


