import numpy as np

from rsna_knee.dicom import (
    centered_consecutive_indices,
    evenly_spaced_indices,
    geometric_position,
    infer_position_laterality,
    patient_center_x,
    robust_uint8,
    should_flip_horizontal,
    should_reverse_slice_order,
    slice_sort_key,
)


def test_geometry_position_and_fallback_order():
    base = {
        "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
        "ImagePositionPatient": [0, 0, 12],
    }
    assert geometric_position(base) == 12.0
    assert slice_sort_key(base)[0] == 0
    assert slice_sort_key({"SliceLocation": "3.5"}) == (1, 3.5)
    assert slice_sort_key({"InstanceNumber": 7}) == (2, 7.0)


def test_percentile_normalization_is_bounded():
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    result = robust_uint8(image, 0, 100)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_even_indices_and_laterality_flip_policy():
    np.testing.assert_array_equal(evenly_spaced_indices(5, 3, 0), [0, 2, 4])
    left = {"ImageLaterality": "L", "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}
    right = {"ImageLaterality": "R", "ImageOrientationPatient": [1, 0, 0, 0, 1, 0]}
    assert should_flip_horizontal(left, "Coronal")
    assert not should_flip_horizontal(right, "Coronal")
    assert not should_flip_horizontal(left, "Sagittal")


def test_position_laterality_and_sagittal_slice_order():
    base = {
        "ImageOrientationPatient": [0, 1, 0, 0, 0, -1],
        "ImagePositionPatient": [-100, -128, -128],
        "PixelSpacing": [1, 1],
        "Rows": 256,
        "Columns": 256,
    }
    assert patient_center_x(base) == -100
    assert infer_position_laterality(base) == "R"
    assert should_reverse_slice_order(base, "Sagittal")
    left = {**base, "ImagePositionPatient": [100, -128, -128]}
    assert infer_position_laterality(left) == "L"
    assert not should_reverse_slice_order(left, "Sagittal")


def test_centered_indices_are_physically_adjacent():
    np.testing.assert_array_equal(centered_consecutive_indices(21, 5), [8, 9, 10, 11, 12])
    np.testing.assert_array_equal(centered_consecutive_indices(2, 5), [0, 0, 0, 1, 1])


def test_image_centre_position_infers_laterality_not_top_left_corner():
    image = {
        "ImageOrientationPatient": [1, 0, 0, 0, 1, 0],
        "ImagePositionPatient": [-150, -100, 0],
        "PixelSpacing": [1, 1],
        "Rows": 201,
        "Columns": 201,
    }
    assert patient_center_x(image) == -50.0
    assert infer_position_laterality(image) == "R"
    image["ImagePositionPatient"] = [50, -100, 0]
    assert patient_center_x(image) == 150.0
    assert infer_position_laterality(image) == "L"
