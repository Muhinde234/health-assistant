# train.py
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_loader import load_data
from preprocessing import preprocess_data, split_data
from config import DATA_URL, MODEL_PATH

def train():
 
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(BASE_DIR, "models", "heart_model.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print(f"Model will be saved to: {model_path}")

   
    df = load_data(DATA_URL)
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)


    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42)
    }

    best_model = None
    best_score = 0

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        score = accuracy_score(y_test, preds)
        print(f"{name} accuracy: {score:.4f}")

        if score > best_score:
            best_model = model
            best_score = score

  
    joblib.dump(best_model, model_path)
    if os.path.exists(model_path):
        print(f"Best model saved successfully at: {model_path}")
    else:
        print("Failed to save model!")

if __name__ == "__main__":
    train()