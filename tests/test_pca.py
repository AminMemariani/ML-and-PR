from unsupervised_learning.pca_dimensionality_reduction import load_data, run_pca


def test_pca_reduction_shape_and_variance():
    X, class_names = load_data()
    components, explained = run_pca(X)

    # Expect two components for each original sample.
    assert components.shape[1] == 2
    assert len(components) == len(X)

    # Explained variance ratios should sum to less than or equal to 1 and be informative.
    assert 0.0 < explained[0] <= 1.0
    assert explained.sum() <= 1.0
    assert explained.sum() > 0.4

    assert len(class_names) == 3

