import os
import joblib
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")


model = joblib.load(MODEL_PATH)

def predict_patient(data):
    """
    Predict heart disease for a single patient.
    data: list of feature values in correct order
    """
    data = np.array(data).reshape(1, -1)
    prediction = model.predict(data)
    return prediction