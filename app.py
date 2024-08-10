import joblib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_model():
    hgbr_model = joblib.load('../models/HistGradientBoostingRegressor_best_model.joblib'),
    kn_model = joblib.load('..models/KNeighborsRegressor_best_model.joblib'),
    lgbm_model = joblib.load('../models/models/LGBMRegressor_best_model.joblib'),
    xgb_model =  joblib.load('../models/models/XGBRegressor_best_model.joblib')

    return hgbr_model, kn_model, lgbm_model, xgb_model  


def load_and_fit_preprocessor(file_path):
    housing_df = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaler.fit(housing_df)

def user_input_features():
    inputs = {
        'SqFtTotLiving' = st.sidebar.number_input('Total Living Area (SqFt)', 0, 20000, 1500),
        'Latitude' = st.sidebar.number_input('Latitude', -90.0, 90.0, 47.0),
        'Longitude' = st.sidebar.number_input('Longitude', -180.0, 180.0, -122.0),
        'SqFt2ndFloor' = st.sidebar.number_input('Second Floor Area (SqFt)', 0, 10000, 0),
        'YrBuilt' = st.sidebar.number_input('Year Built', 1800, 2024, 2000),
        'SqFtFinBasement' = st.sidebar.number_input('Finished Basement Area (SqFt)', 0, 10000, 0),
        'SqFtDeck' = st.sidebar.number_input('Deck Area (SqFt)', 0, 5000, 0),
        'SqFtGarageBasement' = st.sidebar.number_input('Garage Basement Area (SqFt)', 0, 5000, 0),
        'BrickStone' = st.sidebar.selectbox('Brick/Stone Exterior', ['Yes', 'No']),
        'SqFtOpenPorch' = st.sidebar.number_input('Open Porch Area (SqFt)', 0, 5000, 0),
    }

    return pd.DataFrame(inputs, index=[0])

def make_prediction(model, scaler, input_df):
    input_df_transformed = scaler.transform(input_df)
    prediction = model.predict(input_df_transformed)

    return 'Satisfied 😊' if prediction[0] else 'Not Satisfied 😞'

#collect user input
st.sidebar.header('User Input')
input_df = user_input_features()

#display user inputs
st.write('### User Inputs')


models = load_model()
scaler = load_and_fit_preprocessor('data/transformed/new_housing_dataset.csv')


st.title("House Price Prediction App")

st.divider()

st.write("The app uses machine learning for predicting house price with the given features of the house.")

st.divider()

#collect user input
st.sidebar.header('User Input')
input_df = user_input_features()

#display user inputs
st.write('### User Inputs')

for key, value in input_df.iloc[0].items():
    st.write(f'**{key}**: {value}')

model_choice = st.sidebar.selectbox('Choose Model 🤖', list(models.keys()))
selected_model = models[model_choice]
st.write(f'### Selected model: {model_choice}')

#prediction button
if st.button('Predict'):
    try: 
        result = make_prediction(selected_model, scaler, input_df)
        st.success(f'You are {result}')

    except Exception as e:
        st.error(f'Error Occured during prediction. {e}\n Please contact support')