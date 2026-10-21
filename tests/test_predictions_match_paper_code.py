import numpy as np
import pytest

from tests.fixed_input_predictions import COMPONENTS, PAPER_CODE_PREDICTIONS, compute_predictions_on_fixed_inputs


@pytest.fixture(scope="module")
def predictions():
    return compute_predictions_on_fixed_inputs()


@pytest.fixture(scope="module")
def paper_code_predictions():
    return np.load(PAPER_CODE_PREDICTIONS)


def test_same_outputs_as_paper_code(predictions, paper_code_predictions):
    assert set(predictions) == set(paper_code_predictions.files)


@pytest.mark.parametrize("component", COMPONENTS)
def test_predictions_match_paper_code(component, predictions, paper_code_predictions):
    keys = [k for k in paper_code_predictions.files if k.startswith(f"{component}/")]
    assert keys
    for key in keys:
        np.testing.assert_allclose(predictions[key], paper_code_predictions[key], rtol=0, atol=1e-5, err_msg=key)
