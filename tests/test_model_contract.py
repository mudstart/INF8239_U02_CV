import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
from inf8239_u02_cv.data import NUM_CLASSES
from inf8239_u02_cv.models import build_cnn, build_dense

INPUTS = {
    "ceros": np.zeros((2, 28, 28, 1), dtype="float32"),
    "aleatoria": np.random.default_rng(42).random((5, 28, 28, 1), dtype="float32"),
}


@pytest.mark.parametrize("factory", [build_dense, build_cnn], ids=["densa", "cnn"])
@pytest.mark.parametrize("images", INPUTS.values(), ids=INPUTS.keys())
def test_model_output_contract(factory, images):
    probabilities = factory(tf)(images).numpy()
    assert probabilities.shape == (len(images), NUM_CLASSES)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
