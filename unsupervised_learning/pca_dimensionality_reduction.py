"""Principal Component Analysis example for dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the classic Wine dataset from scikit-learn."""

    # `load_wine` returns a Bunch with feature matrix and metadata. Using `as_frame=True`
    # gives a pandas DataFrame, keeping column names for interpretability.
    dataset = load_wine(as_frame=True)
    return dataset.data, dataset.target_names


def run_pca(X):
    """Scale features before applying PCA and return reduced data."""

    # Standardize features so each contributes equally; PCA is sensitive to scale.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA to project data onto first two principal components.
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled)
    # Explained variance ratio tells us how much information each component captures.
    explained = pca.explained_variance_ratio_

    return components, explained


def main() -> None:
    # Load the dataset and class labels for reference.
    X, class_names = load_data()
    # Reduce to two components to facilitate visualization and noise reduction scenarios.
    components, explained = run_pca(X)

    print("First two principal components (first five rows):")
    print(components[:5])

    print("\nExplained variance ratio:")
    for idx, value in enumerate(explained, start=1):
        print(f"  PC{idx}: {value:.3f}")

    print("\nOriginal target classes:", list(class_names))


if __name__ == "__main__":
    main()

