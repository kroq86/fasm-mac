import numpy as np
import pytest

import eml_sr


def test_fit_exp():
    x = np.array([0.1, 0.5, 1.0])
    y = np.exp(x)
    result = eml_sr.fit(x, y, max_depth=2)
    assert result.eml_nodes >= 1
    assert result.mse < 1e-6
    assert "eml" in result.rpn
    preds = result.predict(x)
    np.testing.assert_allclose(preds, y, rtol=1e-6, atol=1e-6)


def test_eml_primitive():
    assert abs(eml_sr.eml(1.0, 1.0) - (np.exp(1.0) - np.log(1.0))) < 1e-9


def test_fit_rejects_bad_method_and_domain():
    x = np.array([0.1, 0.5, 1.0])
    y = np.exp(x)
    with pytest.raises(ValueError):
        eml_sr.fit(x, y, method="nope")
    with pytest.raises(ValueError):
        eml_sr.fit(x, y, domain="nope")


@pytest.mark.slow
def test_fit_poly_depth4():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = x * x + 1.0
    result = eml_sr.fit(x, y, max_depth=4, method="enumerate")
    assert np.isfinite(result.mse)
    assert result.eml_nodes == 4
    preds = result.predict(x)
    # v1 enumerate target: materially better than mean baseline, not exact x^2+1 snap.
    baseline = float(np.mean((y - np.mean(y)) ** 2))
    assert result.mse < baseline
    assert result.mse < 0.2
