import numpy as np
from sklearn.metrics import accuracy_score

from supervised_learning.logistic_regression_demo import (
    evaluate,
    load_data,
    prepare_data,
    summarize_results,
    train_model,
)


def test_data_generation_reproducible():
    first = load_data(seed=1)
    second = load_data(seed=1)
    third = load_data(seed=2)

    # Same seed should yield identical frames; different seed should differ.
    assert first.equals(second)
    assert not first.equals(third)


def test_logistic_workflow_produces_quality_metrics(capsys):
    df = load_data(seed=0)
    X_train, X_test, y_train, y_test, _scaler, feature_names = prepare_data(df)

    model = train_model(X_train, y_train)
    summarize_results(model, feature_names)
    evaluate(model, X_test, y_test)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    assert accuracy > 0.7

    captured = capsys.readouterr().out
    assert "Logistic regression coefficients" in captured
    assert "Classification report" in captured

