"""Logistic regression example modeling a binary health outcome.

Run the script with:
    python logistic_regression_demo.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic medical screening dataset."""

    # Use a dedicated random generator for reproducible results.
    rng = np.random.default_rng(seed)

    # Draw feature values from simple distributions so the relationship is interpretable.
    size = 300
    ages = rng.normal(50, 9, size).round(1)
    tumor_sizes = rng.normal(2.7, 0.6, size).round(2)
    biomarker = rng.normal(7.5, 1.2, size).round(2)

    # Build latent logits that combine the features, then convert them to probabilities via the sigmoid function.
    logits = 0.06 * ages + 0.9 * tumor_sizes + 0.4 * biomarker - 10
    probabilities = 1 / (1 + np.exp(-logits))
    # Sample binary labels (0 = benign, 1 = malignant) based on those probabilities.
    labels = rng.binomial(n=1, p=probabilities)

    return pd.DataFrame(
        {
            "age": ages,
            "tumor_size_cm": tumor_sizes,
            "biomarker_score": biomarker,
            "malignant": labels,
        }
    )


def prepare_data(df: pd.DataFrame):
    """Split the dataset, scaling features to stabilize training."""

    # Separate predictors from the target column.
    X = df.drop(columns="malignant")
    y = df["malignant"]

    # Stratify ensures class proportions remain consistent across train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    # Standardize features to zero mean / unit variance. Logistic regression converges more reliably with scaled data.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns.tolist()


def train_model(X_train: np.ndarray, y_train: pd.Series) -> LogisticRegression:
    """Train logistic regression with balanced class weights."""

    # class_weight="balanced" compensates for any mild class imbalance by
    # reweighting samples inversely proportional to class frequency.
    model = LogisticRegression(class_weight="balanced", max_iter=200, random_state=42)
    model.fit(X_train, y_train)
    return model


def summarize_results(model: LogisticRegression, feature_names: list[str]) -> None:
    """Print coefficient-based feature importance."""

    # Coefficients indicate how each feature changes the log-odds of malignancy.
    print("Logistic regression coefficients (log-odds impact):")
    for name, coef in zip(feature_names, model.coef_[0], strict=True):
        print(f"  {name:>15}: {coef:>6.3f}")


def evaluate(model: LogisticRegression, X_test: np.ndarray, y_test: pd.Series) -> None:
    """Report confusion matrix and precision/recall/F1."""

    # Predict binary labels for the hold-out set.
    predictions = model.predict(X_test)
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    # Precision/recall/F1 helps examine trade-offs when false positives/negatives carry different costs.
    print(classification_report(y_test, predictions, digits=3))


def main() -> None:
    # Generate synthetic patient data.
    df = load_data()
    # Split and scale to prepare inputs for logistic regression.
    X_train, X_test, y_train, y_test, _scaler, feature_names = prepare_data(df)
    # Fit the classifier on the training data.
    model = train_model(X_train, y_train)

    summarize_results(model, feature_names)
    # Evaluate generalization by summoning predictions on unseen examples.
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()

