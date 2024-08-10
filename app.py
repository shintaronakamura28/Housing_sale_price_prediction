import joblib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

hgbr_model = joblib.load('../models/HistGradientBoostingRegressor_best_model.joblib')
kn_model = joblib.load('..models/KNeighborsRegressor_best_model.joblib')
lgbm_model = joblib.load('../models/models/LGBMRegressor_best_model.joblib')
xgb_model =  joblib.load('../models/models/XGBRegressor_best_model.joblib')


st.title("House Price Prediction App")

st.divider()

st.write("The app uses machine learning for predicting house price with the given features of the house.")

st.divider()


def user_input_features():
    inputs = {
    'SqFtTotLiving' = st.sidebar.number_input('Total Living Area (SqFt)', 0, 20000, 1500)
    'Latitude' = st.sidebar.number_input('Latitude', -90.0, 90.0, 47.0)
    'Longitude' = st.sidebar.number_input('Longitude', -180.0, 180.0, -122.0)
    'SqFt2ndFloor' = st.sidebar.number_input('Second Floor Area (SqFt)', 0, 10000, 0)
    'YrBuilt' = st.sidebar.number_input('Year Built', 1800, 2024, 2000)
    'SqFtFinBasement' = st.sidebar.number_input('Finished Basement Area (SqFt)', 0, 10000, 0)
    'SqFtDeck' = st.sidebar.number_input('Deck Area (SqFt)', 0, 5000, 0)
    'SqFtGarageBasement' = st.sidebar.number_input('Garage Basement Area (SqFt)', 0, 5000, 0)
    'BrickStone' = st.sidebar.selectbox('Brick/Stone Exterior', ['Yes', 'No'])
    'SqFtOpenPorch' = st.sidebar.number_input('Open Porch Area (SqFt)', 0, 5000, 0)
    }

st.divider()

X = [(SqFtTotLiving, Latitude, Longitude, SqFt2ndFloor, 
      YrBuilt, SqFtFinBasement, SqFtDeck, SqFtGarageBasement, BrickStone, SqFtOpenPorch )]



# Function to load the model using joblib
def load_model(model_name, directory=Path('../models')):
    model_path = directory / f'{model_name}_best_model.joblib'
    model = joblib.load(model_path)
    print(f'Model {model_name} loaded from {model_path}')
    return model