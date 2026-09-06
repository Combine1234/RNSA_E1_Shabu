import numpy as np

from rsna_knee.folds import grouped_multilabel_folds


def test_groups_never_cross_folds_and_result_is_deterministic():
    rng = np.random.default_rng(4)
    targets = rng.random((30, 4))
    groups = np.array([f"g{i // 2}" for i in range(30)])
    first = grouped_multilabel_folds(targets, groups, n_folds=5, seed=7)
    second = grouped_multilabel_folds(targets, groups, n_folds=5, seed=7)
    np.testing.assert_array_equal(first, second)
    for group in np.unique(groups):
        assert len(np.unique(first[groups == group])) == 1
    assert set(first) == set(range(5))


def test_large_soft_label_folds_are_balanced():
    rng = np.random.default_rng(11)
    targets = rng.beta(2, 5, size=(1000, 12))
    groups = np.array([f"g{i}" for i in range(len(targets))])
    folds = grouped_multilabel_folds(targets, groups, n_folds=5, seed=19)
    counts = np.bincount(folds, minlength=5)
    prevalences = np.stack([targets[folds == fold].mean(axis=0) for fold in range(5)])
    assert counts.max() - counts.min() <= 5
    assert np.max(np.ptp(prevalences, axis=0)) < 0.015
