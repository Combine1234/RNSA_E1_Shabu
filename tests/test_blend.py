import numpy as np

from rsna_knee.blend import rank_fuse, select_target_weights


def test_rank_fuse_and_promotion_gate():
    truth = np.array([0, 0, 0, 1, 1, 1], dtype=bool)
    weak = np.array([0.2, 0.1, 0.8, 0.3, 0.7, 0.9])
    strong = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    predictions = np.column_stack([weak, strong])
    weights, diagnostics = select_target_weights(
        predictions, truth, np.ones(6, bool), np.ones(6, bool), step=0.25, minimum_gain=0.0
    )
    assert weights[1] >= weights[0]
    fused = rank_fuse(predictions, weights)
    assert fused.shape == truth.shape
    assert diagnostics["best_single_pseudo_auc"] == 1.0

