import pandas as pd
from sklearn.metrics import mean_absolute_error

from supervised_learning.linear_regression_demo import (
    describe_model,
    evaluate,
    load_data,
    train_model,
)


def test_load_data_structure():
    data = load_data()
    assert isinstance(data, pd.DataFrame)
    assert set(data.columns) == {"sq_ft", "bedrooms", "age_years", "price"}
    assert len(data) >= 8


def test_training_and_evaluation_outputs(capsys):
    data = load_data()
    features = data.drop(columns="price")
    target = data["price"]

    model, X_test, y_test = train_model(features, target)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    # With this synthetic dataset, the model should be fairly accurate.
    assert mae < 40000

    describe_model(model, list(features.columns))
    evaluate(model, X_test, y_test)

    captured = capsys.readouterr().out
    assert "Mean Absolute Error" in captured
    assert "Learned intercept" in captured

