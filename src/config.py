import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"

MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.pkl")