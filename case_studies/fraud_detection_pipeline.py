"""Fraud detection mini case study with a tree-based model pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(seed: int = 0) -> pd.DataFrame:
    """Create a synthetic credit-card transaction dataset."""

    # Generate synthetic transactions so we can control the fraud rate and contributing factors.
    rng = np.random.default_rng(seed)
    size = 1500

    df = pd.DataFrame(
        {
            "transaction_amount": rng.gamma(shape=2.0, scale=120.0, size=size),
            "transaction_hour": rng.integers(0, 24, size=size),
            "merchant_category": rng.choice(["electronics", "groceries", "fashion", "travel"], size=size),
            "device_type": rng.choice(["mobile", "desktop"], p=[0.65, 0.35], size=size),
        }
    )

    # Begin with a small base fraud rate (3%) and add interpretable feature contributions.
    base_rate = 0.03
    amount_factor = np.clip((df["transaction_amount"] - 250) / 400, 0, 1)
    night_factor = ((df["transaction_hour"] >= 22) | (df["transaction_hour"] <= 5)).astype(int) * 0.4
    device_factor = (df["device_type"] == "mobile").astype(int) * 0.2
    category_factor = df["merchant_category"].map({"electronics": 0.25, "groceries": 0.0, "fashion": 0.15, "travel": 0.3})

    # Combine the effects to control the probability of fraud, then sample binary outcomes.
    fraud_probability = base_rate + amount_factor + night_factor + device_factor + category_factor
    fraud_probability = fraud_probability.clip(0, 0.95)
    df["is_fraud"] = rng.binomial(n=1, p=fraud_probability)

    return df


def build_pipeline(
    n_estimators: int = 300,
    max_depth: int | None = None,
    param_grid: dict[str, list] | None = None,
    cv: int = 3,
) -> GridSearchCV:
    """Create preprocessing + random forest pipeline wrapped in grid search."""

    # Separate numeric and categorical features to apply appropriate preprocessing in a ColumnTransformer.
    numeric_features = ["transaction_amount", "transaction_hour"]
    categorical_features = ["merchant_category", "device_type"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Random forest handles nonlinear feature interactions and class_weight addresses imbalance.
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Combine preprocessing and model into a single pipeline for cleaner training/validation.
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

    # Hyperparameters to tune: tree depth and minimum samples needed to split a node.
    search_space = param_grid or {
        "model__max_depth": [None, 8, 12],
        "model__min_samples_split": [2, 8, 16],
    }

    grid = GridSearchCV(
        pipeline,
        param_grid=search_space,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )

    return grid


def evaluate(model: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Print ROC-AUC and class-specific metrics."""

    # Obtain predicted probabilities for ROC-AUC and hard predictions for class report.
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    auc = roc_auc_score(y_test, probabilities)
    print("ROC-AUC:", f"{auc:.3f}")
    print("\nClassification report:")
    # Classification report highlights recall, important in fraud detection where missing fraud is costly.
    print(classification_report(y_test, predictions, digits=3))


def main() -> None:
    # Synthesize labeled data and split into training/hold-out sets.
    df = load_data()
    X = df.drop(columns="is_fraud")
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    # Build the preprocessing/model pipeline and perform grid search on the training data.
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("Best parameters:", model.best_params_)
    # Evaluate the tuned pipeline on the hold-out set.
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()

