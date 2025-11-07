# Machine Learning Core Concepts by Example

Hands-on Python snippets to revisit the foundations of machine learning and pattern recognition. Each folder contains a focused script you can run from the command line to see the full workflow end to end. The examples keep datasets deliberately small so you can inspect raw values and understand *why* a model behaves the way it does.

## Prerequisites

- Python 3.10 or newer
- `pip` for dependency management
- Optional: JupyterLab or VS Code if you prefer running the scripts inside notebooks

All required third-party packages live in `requirements.txt`.

## Project Layout

- `supervised_learning/`
  - `linear_regression_demo.py`: predict housing prices with linear regression, inspect coefficients, and evaluate with MAE/R².
  - `logistic_regression_demo.py`: classify synthetic medical screening outcomes with logistic regression and interpret odds.
- `unsupervised_learning/`
  - `kmeans_clustering.py`: cluster synthetic customer profiles and measure separation with the silhouette score.
  - `pca_dimensionality_reduction.py`: compress the wine dataset to two principal components and inspect explained variance.
- `model_evaluation/`
  - `model_validation_basics.py`: illustrate hold-out evaluation, cross-validation, hyperparameter search, and learning curves.
- `case_studies/`
  - `fraud_detection_pipeline.py`: end-to-end pipeline for synthetic credit-card fraud detection using preprocessing + random forest.

## Getting Started

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run any script** to explore the concept, e.g.:

   ```bash
   python supervised_learning/linear_regression_demo.py
   ```

Each script prints intermediate insights so you can understand how data preparation, model fitting, and evaluation connect.

## What Each Folder Demonstrates

### Supervised Learning

Run the scripts inside `supervised_learning/` to see full regression and classification pipelines:

- `linear_regression_demo.py` highlights feature scaling decisions, coefficient interpretation, and R²/MAE evaluation on a toy housing dataset.
- `logistic_regression_demo.py` demonstrates stratified splitting, feature standardization, and classification metrics (precision, recall, F1) for a synthetic medical screening scenario.

### Unsupervised Learning

- `kmeans_clustering.py` generates three customer segments, then reports centroid locations and silhouette score to judge how well the clusters separate.
- `pca_dimensionality_reduction.py` loads the classic wine dataset, applies standardization before PCA, and prints the variance explained by the first two components, preparing you for 2D visualization.

### Model Evaluation

- `model_evaluation/model_validation_basics.py` walks through hold-out validation, K-fold cross-validation, hyperparameter tuning with `GridSearchCV`, and how performance evolves via a learning curve.

### Case Study

- `case_studies/fraud_detection_pipeline.py` builds a preprocessing + random-forest pipeline for synthetic credit-card fraud. It uses column transformers to handle mixed numeric/categorical data, searches over hyperparameters, and reports ROC-AUC alongside class metrics.

## Suggested Experiments

- Visualize predictions vs. actual values by adding `matplotlib` plots to the regression script.
- Try alternative clustering algorithms (`DBSCAN`, `AgglomerativeClustering`) on the same data to compare behavior.
- Modify the fraud dataset generator to add new signals (e.g., transaction velocity) and observe how the model’s feature importances change.
- Wrap scripts in notebooks and add markdown commentary to turn them into study notes.

## Next Steps

- Modify randomness seeds or add noise to observe robustness.
- Swap models (e.g., try `RandomForestRegressor` in the regression example) and compare metrics.
- Extend the fraud case study with additional engineered features or alternative classifiers.

