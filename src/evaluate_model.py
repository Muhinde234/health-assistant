# evaluate_model.py
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_PATH, DATA_URL

# Import paths from config
from config import MODEL_PATH
from data_loader import load_data
from preprocessing import preprocess_data

# -------------------------------
# Load the trained model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train your model first.")

model = joblib.load(MODEL_PATH)
print(f"Loaded model from {MODEL_PATH}: {type(model).__name__}")

# -------------------------------
# Feature importance function
# -------------------------------
def feature_importance(model, features):
    """
    Plot and return feature importance if available.
    
    Args:
        model: Trained model (RandomForestClassifier or similar)
        features: list of feature names
    
    Returns:
        pd.DataFrame of feature importance
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"The model of type {type(model).__name__} does not support feature_importances_."
        )

    importance = model.feature_importances_

    df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=True)

    # Plot
    plt.figure(figsize=(10,6))
    sns.barplot(x="importance", y="feature", data=df, palette="viridis")

    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")

    plt.tight_layout()
    plt.show()

    return df


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    # Load dataset
    df =  load_data(DATA_URL)

    # Preprocess
    X, y = preprocess_data(df)

    # Get feature names
    feature_names = X.columns.tolist()

    try:
        df_importance = feature_importance(model, feature_names)
        print(df_importance)

    except AttributeError as e:
        print(e)
        print("Feature importance is only available for tree-based models (e.g., RandomForestClassifier).")