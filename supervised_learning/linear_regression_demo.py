"""Linear regression example exploring a small housing-style dataset.

Run this script directly to see the workflow:
    python linear_regression_demo.py
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def load_data() -> pd.DataFrame:
    """Create a tiny synthetic housing dataset."""

    # Manually construct a tidy DataFrame so the raw numbers remain easy to inspect.
    # Each column represents a feature except `price`, which is the target we aim to predict.
    return pd.DataFrame(
        {
            "sq_ft": [850, 900, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3200],
            "bedrooms": [2, 2, 3, 3, 4, 4, 4, 5, 5, 6],
            "age_years": [40, 35, 30, 20, 15, 10, 8, 5, 4, 3],
            "price": [185000, 192000, 235000, 268000, 320000, 355000, 375000, 430000, 455000, 510000],
        }
    )


def train_model(features: pd.DataFrame, target: pd.Series) -> tuple[LinearRegression, pd.DataFrame, pd.Series]:
    """Split data, fit a linear regression model, and return the components."""

    # Split into train and test subsets to estimate generalization performance.
    # We use a fixed random seed for reproducibility of the example output.
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    # Initialize a plain linear regression model (ordinary least squares) and fit it.
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_test, y_test


def describe_model(model: LinearRegression, feature_names: list[str]) -> None:
    """Print learned coefficients as human-readable insight."""

    # Zip together feature names with their corresponding coefficients so the
    # impact of each feature (slope) becomes easy to interpret.
    if len(feature_names) != len(model.coef_):
        raise ValueError("Feature names and coefficients length mismatch.")
    coef_by_feature = dict(zip(feature_names, model.coef_))
    intercept = model.intercept_

    # The intercept is the baseline prediction when all features are zero.
    print("Learned intercept (baseline price):", f"{intercept:,.0f}")
    for name, coef in coef_by_feature.items():
        # Each coefficient describes how much the predicted price changes when
        # that feature increases by one unit, holding other features constant.
        print(f"  {name}: {coef:,.2f} per unit")


def evaluate(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Compute regression metrics on the hold-out set."""

    # Generate predictions for the unseen test data.
    predictions = model.predict(X_test)
    # Mean absolute error (MAE) summarizes the average absolute difference between
    # actual prices and predicted prices.
    mae = mean_absolute_error(y_test, predictions)
    # R² indicates how much variance in the target is explained by the model.
    r2 = r2_score(y_test, predictions)

    print("\nEvaluation on test set:")
    print("  Mean Absolute Error:", f"{mae:,.0f}")
    print("  R^2 Score:", f"{r2:.3f}")


def main() -> None:
    # Load the dataset containing features and the target `price`.
    data = load_data()
    # Separate the features (independent variables) from the target variable.
    features = data.drop(columns="price")
    target = data["price"]

    # Train the model and retain the test split for evaluation.
    model, X_test, y_test = train_model(features, target)

    print("Feature importances inferred by linear regression:\n")
    # Inspect learned coefficients to understand the model's reasoning.
    describe_model(model, list(features.columns))
    # Assess how well the model generalizes to unseen data.
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()

