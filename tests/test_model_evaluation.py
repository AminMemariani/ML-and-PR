import numpy as np

from model_evaluation.model_validation_basics import (
    cross_validation_example,
    holdout_evaluation_example,
    hyperparameter_tuning_example,
    learning_curve_example,
    load_data,
)


def test_cross_validation_outputs(capsys):
    X, y = load_data()
    cross_validation_example(X, y)
    captured = capsys.readouterr().out
    assert "Cross-validation scores" in captured


def test_hyperparameter_search_reports_best_score(capsys):
    X, y = load_data()
    hyperparameter_tuning_example(X, y)
    captured = capsys.readouterr().out
    assert "Best alpha" in captured
    assert "RMSE" in captured


def test_learning_curve_and_holdout(capsys):
    X, y = load_data()
    learning_curve_example(X, y)
    holdout_evaluation_example(X, y)
    captured = capsys.readouterr().out
    assert "Learning curve" in captured
    assert "Hold-out RMSE" in captured

