import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("data/Week_2/model/model.joblib")
    data = pd.read_csv("data/Week_2/train_data.csv")

    X = data.drop("target", axis=1)
    y = data["target"]
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    assert acc > 0.85, f"Model accuracy too low: {acc}"
