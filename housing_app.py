import joblib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_model():
    hgbr_model = joblib.load('../models/HistGradientBoostingRegressor_best_model.joblib')
    kn_model = joblib.load('../models/KNeighborsRegressor_best_model.joblib')
    lgbm_model = joblib.load('./models/LGBMRegressor_best_model.joblib')
    xgb_model = joblib.load('./models/XGBRegressor_best_model.joblib')

    return {
        'HistGradientBoostingRegressor': hgbr_model,
        'KNeighborsRegressor': kn_model,
        'LGBMRegressor': lgbm_model,
        'XGBRegressor': xgb_model
    }


def load_and_fit_preprocessor(file_path):
    housing_df = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaler.fit(housing_df)
    return scaler

def user_input_features():
    inputs = {
        'SqFtTotLiving': st.sidebar.number_input('Total Living Area (SqFt)', 0, 20000, 0),
        'Latitude': st.sidebar.number_input('Latitude', -90.0, 90.0, 47.0),
        'Longitude': st.sidebar.number_input('Longitude', -180.0, 180.0, -122.0,),
        'SqFt2ndFloor': st.sidebar.number_input('Second Floor Area (SqFt)', 0, 10000, 0),
        'YrBuilt': st.sidebar.number_input('Year Built', 0, 2019, 0),
        'SqFtFinBasement': st.sidebar.number_input('Finished Basement Area (SqFt)', 0, 10000, 0),
        'SqFtDeck': st.sidebar.number_input('Deck Area (SqFt)', 0, 5000, 0),
        'SqFtGarageBasement': st.sidebar.number_input('Garage Basement Area (SqFt)', 0, 5000, 0),
        'BrickStone': st.sidebar.number_input('Brick/Stone Exterior',  0, 500 ,0),
        'SqFtOpenPorch': st.sidebar.number_input('Open Porch Area (SqFt)', 0, 5000, 0,),
    }
    return pd.DataFrame(inputs, index=[0])

def make_prediction(model, scaler, input_df):
    print("Input Data before Scaling:")
    print(input_df)
    input_df_transformed = scaler.transform(input_df)
    print("Input Data after Scaling:")
    print(input_df_transformed)
    prediction = model.predict(input_df_transformed)
    return prediction[0]

models = load_model()
scaler = load_and_fit_preprocessor('data/transformed/new_housing_dataset.csv')



def main():
    st.title("House Price Prediction App")
    st.write("The app uses machine learning to predict house prices based on various features of the house.")
    st.divider()

    model_choice = st.selectbox('Choose Model 🤖', list(models.keys()))
    selected_model = models[model_choice]
    st.write(f'### Selected model: {model_choice}')

    st.sidebar.header('User Input')
    input_df = user_input_features()

    st.write('### User Inputs')
    for key, value in input_df.iloc[0].items():
        st.write(f'**{key}**: {value}')

    #prediction button
    if st.button('Predict'):
        prediction = make_prediction(selected_model, scaler, input_df)
        st.write(f'### Predicted Property Price: ${prediction:.2f}')

if __name__ == '__main__':
    main()