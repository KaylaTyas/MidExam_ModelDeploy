# Kayla Masayuningtyas - 2602141871 - LB09

import streamlit as st
import joblib
import numpy as np

model = joblib.load('TrainedModel.pkl')

def main():
    st.title('Machine Learning Model Deployment')

    # Add user input components for 11 features
    unnamed_0 = st.slider('Un named', min_value=0, max_value=41258, value=1)
    credit_score = st.slider('credit score', min_value=350, max_value=850, value=10)
    geography = st.radio('Pick one Goegraphy:', ['0','1', '2'])
    gender = st.radio('Pick one Gender:', ['0','1'])
    age = st.slider('age', min_value=18, max_value=92, value=1)
    Tenure = st.slider('Tenure', min_value=0, max_value=10, value=1)
    balance = st.slider('balance', min_value=0.0, max_value=250899.0, value=200.0)
    numoff_product = st.slider('numoff product', min_value=1, max_value=4, value=1)
    Hascr_Card = st.slider('Hascr Card', min_value=0, max_value=1, value=1)
    isactive_members = st.selectbox('Select is active members', [0, 1])
    estimated_salary = st.slider('Estimated Salary', min_value=11.6, max_value=199992.5, value=0.5)

    
    if st.button('Make Prediction'):
        features = [unnamed_0, credit_score,geography,gender,age, Tenure, balance, numoff_product, Hascr_Card, isactive_members, estimated_salary]
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()