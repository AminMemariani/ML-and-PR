from sklearn.model_selection import train_test_split

from case_studies.fraud_detection_pipeline import build_pipeline, evaluate, load_data


def test_fraud_pipeline_training_and_evaluation(capsys):
    df = load_data(seed=123)
    assert {"transaction_amount", "transaction_hour", "merchant_category", "device_type", "is_fraud"} <= set(df.columns)

    X = df.drop(columns="is_fraud")
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    grid = build_pipeline(
        n_estimators=100,
        param_grid={"model__max_depth": [None], "model__min_samples_split": [2, 4]},
        cv=2,
    )

    grid.fit(X_train, y_train)
    assert "model__max_depth" in grid.best_params_

    evaluate(grid, X_test, y_test)
    captured = capsys.readouterr().out
    assert "ROC-AUC" in captured
    assert "Classification report" in captured

