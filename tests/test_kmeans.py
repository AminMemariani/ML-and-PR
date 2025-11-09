import numpy as np

from unsupervised_learning.kmeans_clustering import evaluate, load_data, train_model


def test_kmeans_clustering_structure(capsys):
    X = load_data(seed=123)
    assert isinstance(X, np.ndarray)
    assert X.shape[1] == 2

    model = train_model(X, n_clusters=3)
    evaluate(model, X)

    assert model.cluster_centers_.shape == (3, 2)
    assert set(model.labels_) == {0, 1, 2}

    captured = capsys.readouterr().out
    assert "Cluster centers" in captured
    assert "Silhouette score" in captured

