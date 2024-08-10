import joblib
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Function to load the model using joblib
def load_model(model_name, directory=Path('../models')):
    model_path = directory / f'{model_name}_best_model.joblib'
    model = joblib.load(model_path)
    print(f'Model {model_name} loaded from {model_path}')
    return model