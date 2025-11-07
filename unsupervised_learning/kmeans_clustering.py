"""K-Means clustering example using a synthetic customer segmentation dataset."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score


def load_data(seed: int = 42) -> np.ndarray:
    """Generate blob-like clusters mimicking customer groups."""

    # `make_blobs` generates isotropic Gaussian clusters, perfect for illustrating K-Means behavior.
    features, _ = make_blobs(
        n_samples=400,
        centers=[(-5, -2), (0, 0), (5, 5)],
        cluster_std=[1.2, 1.0, 1.5],
        random_state=seed,
    )
    return features


def train_model(X: np.ndarray, n_clusters: int = 3) -> KMeans:
    """Fit K-Means and return the trained estimator."""

    # Increase `n_init` so K-Means runs multiple initializations and picks the best result.
    model = KMeans(n_clusters=n_clusters, n_init=15, random_state=0)
    model.fit(X)
    return model


def evaluate(model: KMeans, X: np.ndarray) -> None:
    """Print cluster centroids and silhouette score."""

    # Access cluster assignments inferred by the model.
    labels = model.labels_
    # Silhouette score measures cohesion vs. separation; closer to 1 means well-separated clusters.
    # It requires at least 2 clusters and at most n_samples - 1 clusters; satisfied here.
    score = silhouette_score(X, labels)

    print("Cluster centers:")
    for idx, center in enumerate(model.cluster_centers_, start=1):
        print(f"  Cluster {idx}: {center}")

    print("\nSilhouette score:", f"{score:.3f}")


def main() -> None:
    # Generate synthetic two-dimensional data with three cluster centers.
    X = load_data()
    # Fit K-Means to discover latent clusters.
    model = train_model(X)
    # Display centroid positions and the silhouette score to assess quality.
    evaluate(model, X)


if __name__ == "__main__":
    main()

