import pandas as pd
from ucimlrepo import fetch_ucirepo

def load_data(_):
    heart_disease = fetch_ucirepo(id=45)

    X = heart_disease.data.features
    y = heart_disease.data.targets

    df = pd.concat([X, y], axis=1)

    return df