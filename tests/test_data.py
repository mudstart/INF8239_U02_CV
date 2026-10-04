import numpy as np
import pytest

from inf8239_u02_cv.data import (
    CLASS_NAMES,
    NUM_CLASSES,
    normalize_images,
    validate_images,
)


def valid_batch(n=3):
    return np.full((n, 28, 28, 1), 0.5, dtype="float32"), np.arange(n, dtype=np.uint8)


def test_normalize_adds_channel_and_scales():
    raw = np.array([[[0] * 28] * 28, [[255] * 28] * 28], dtype=np.uint8)
    result = normalize_images(raw)
    assert result.shape == (2, 28, 28, 1)
    assert result.min() == 0
    assert result.max() == 1


def test_validate_accepts_valid_batch():
    validate_images(*valid_batch())


def test_validate_rejects_wrong_shape():
    with pytest.raises(ValueError, match="Forma inesperada"):
        validate_images(np.zeros((2, 28, 28)), np.zeros(2))


@pytest.mark.parametrize("value", [255.0, -0.1, np.nan, np.inf])
def test_validate_rejects_values_outside_unit_range(value):
    images, labels = valid_batch()
    images[0, 0, 0, 0] = value
    with pytest.raises(ValueError, match="normalizadas entre 0 y 1"):
        validate_images(images, labels)


def test_there_are_ten_named_classes():
    assert NUM_CLASSES == 10
    assert len(set(CLASS_NAMES)) == NUM_CLASSES


@pytest.mark.parametrize("label", [-1, NUM_CLASSES])
def test_validate_rejects_labels_outside_classes(label):
    images, _ = valid_batch(2)
    with pytest.raises(ValueError, match="Etiquetas fuera"):
        validate_images(images, np.array([0, label]))


def test_validate_rejects_non_integer_labels():
    images, _ = valid_batch(2)
    with pytest.raises(ValueError, match="enteras"):
        validate_images(images, np.array([0.0, 1.0]))
