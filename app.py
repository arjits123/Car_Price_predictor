import streamlit as st
import pickle


pipe = pickle.load(open('LinearRegressionModel.pkl','rb'))
car_df = pickle.load(open('car_df.pkl','rb'))
st.title("Car Price Predictor")


#Company
st.selectbox('Select the Brand', car_df['company'].unique())

# Model
st.selectbox('Select the Model', car_df['name'].unique())

# Year
st.selectbox('Select Year of Purchase', car_df['year'].unique())

#Fuel Type
st.selectbox('Select Fuel type', car_df['fuel_type'].unique())

# KMs Driven
st.slider('kms Driven', min_value = car_df['kms_driven'].min(), max_value = car_df['kms_driven'].max(), step=1)


if st.button('Predict'):
    pass