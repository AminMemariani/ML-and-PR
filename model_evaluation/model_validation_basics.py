"""Utility script highlighting common model evaluation techniques."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, learning_curve


def load_data():
    # `load_diabetes` provides a regression dataset with continuous targets; return_X_y keeps it NumPy-based.
    dataset = load_diabetes(return_X_y=True)
    return dataset


def cross_validation_example(X: np.ndarray, y: np.ndarray) -> None:
    """Showcase k-fold cross-validation with a Ridge regressor."""

    # Ridge regression adds L2 regularization to mitigate overfitting.
    model = Ridge(alpha=1.0, random_state=42)
    # KFold with shuffling creates five different train/test splits to average performance.
    scores = cross_val_score(model, X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42))

    print("Cross-validation scores:", np.round(scores, 3))
    print("Average +/- std:", f"{scores.mean():.3f} +/- {scores.std():.3f}")


def hyperparameter_tuning_example(X: np.ndarray, y: np.ndarray) -> None:
    """Perform a simple grid search for the regularization strength."""

    # Tune the `alpha` hyperparameter to see how regularization changes performance.
    param_grid = {"alpha": [0.1, 1.0, 10.0, 100.0]}
    model = Ridge(random_state=42)
    grid = GridSearchCV(model, param_grid=param_grid, scoring="neg_mean_squared_error", cv=5)
    grid.fit(X, y)

    # Convert the best (negative) MSE score back into RMSE for interpretability.
    best_alpha = grid.best_params_["alpha"]
    best_rmse = (-grid.best_score_) ** 0.5
    print("Best alpha from grid search:", best_alpha)
    print("Corresponding RMSE:", f"{best_rmse:.3f}")


def learning_curve_example(X: np.ndarray, y: np.ndarray) -> None:
    """Illustrate how performance changes with more training data."""

    # Learning curves reveal bias vs. variance characteristics of the model.
    model = Ridge(alpha=1.0, random_state=42)
    train_sizes, train_scores, test_scores = learning_curve(
        model,
        X,
        y,
        train_sizes=np.linspace(0.2, 1.0, 5),
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
    )

    print("Learning curve (train sizes vs. mean test score):")
    mean_scores = test_scores.mean(axis=1)
    for size, score in zip(train_sizes, mean_scores):
        # `size` returns actual sample counts because we specified absolute sizes.
        print(f"  {int(size)} samples -> score {score:.3f}")


def holdout_evaluation_example(X: np.ndarray, y: np.ndarray) -> None:
    """Train/test split evaluation to illustrate RMSE."""

    # Manually create a randomized 80/20 split using NumPy for transparency.
    rng = np.random.default_rng(42)
    indices = rng.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print("Hold-out RMSE:", f"{rmse:.3f}")


def main() -> None:
    # Load the regression dataset once, then reuse for each evaluation technique.
    X, y = load_data()

    print("=== Hold-out evaluation ===")
    holdout_evaluation_example(X, y)

    print("\n=== Cross-validation ===")
    cross_validation_example(X, y)

    print("\n=== Hyperparameter search ===")
    hyperparameter_tuning_example(X, y)

    print("\n=== Learning curve ===")
    learning_curve_example(X, y)


if __name__ == "__main__":
    main()

