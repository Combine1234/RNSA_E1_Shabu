import numpy as np

from rsna_knee.metrics import average_ranks, binary_auc, macro_auc


def test_average_ranks_handles_ties():
    actual = average_ranks(np.array([10.0, 20.0, 20.0, 40.0]))
    np.testing.assert_allclose(actual, [0.0, 0.5, 0.5, 1.0])


def test_binary_auc_perfect_and_reversed():
    truth = np.array([0, 0, 1, 1])
    assert binary_auc(truth, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert binary_auc(truth, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_macro_auc_respects_mask():
    truth = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    score = np.array([[0.1, 0.9], [0.8, 0.2], [0.2, 0.8], [0.7, 0.1]])
    mask = np.ones_like(truth, dtype=bool)
    value, per_target = macro_auc(truth, score, mask)
    assert value == 1.0
    np.testing.assert_array_equal(per_target, [1.0, 1.0])

