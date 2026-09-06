import numpy as np

from rsna_knee.dataset import BlockShuffleSampler, select_tta_window


def test_tta_offsets_select_three_of_eleven_slices():
    images = np.arange(11).reshape(1, 11, 1, 1)
    valid = np.ones((1, 11), dtype=bool)
    left, _ = select_tta_window(images, valid, -2)
    center, _ = select_tta_window(images, valid, 0)
    right, _ = select_tta_window(images, valid, 2)
    np.testing.assert_array_equal(left.ravel(), np.arange(2, 5))
    np.testing.assert_array_equal(center.ravel(), np.arange(4, 7))
    np.testing.assert_array_equal(right.ravel(), np.arange(6, 9))


def test_block_shuffle_is_complete_deterministic_and_locally_sequential():
    first = list(BlockShuffleSampler(19, block_size=4, seed=7))
    second = list(BlockShuffleSampler(19, block_size=4, seed=7))
    assert first == second
    assert sorted(first) == list(range(19))
    positions = {value: index for index, value in enumerate(first)}
    for start in range(0, 19, 4):
        block = list(range(start, min(start + 4, 19)))
        assert [positions[value] for value in block] == list(
            range(positions[block[0]], positions[block[0]] + len(block))
        )
