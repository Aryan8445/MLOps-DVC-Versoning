import pandas as pd

def test_data_validity():
    data = pd.read_csv("data/Week_2/train_data/data.csv")
    assert data.isnull().sum().sum() == 0, "Data has missing values!"
    expected_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    for col in expected_cols:
        assert col in data.columns, f"Missing column: {col}"
