import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_data(df):

    # Replace ? with NaN
    df = df.replace("?", pd.NA)

    # Convert all columns to numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop rows with missing values
    df = df.dropna()

    # Target
    y = df["num"]

    # Features
    X = df.drop(columns=["num"])

    return X, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test