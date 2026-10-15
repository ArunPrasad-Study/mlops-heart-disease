import pytest
import pandas as pd
import joblib
import os

def test_data_shape():
    """Test if cleaned data has the expected number of columns."""
    df = pd.read_csv('data/heart_cleaned.csv')
    assert df.shape[1] == 14, "Dataset should have 14 columns"

def test_model_loads():
    """Test if the saved model can be loaded successfully."""
    assert os.path.exists('models/best_model.pkl'), "Model file not found"
    model = joblib.load('models/best_model.pkl')
    assert model is not None, "Failed to load model"