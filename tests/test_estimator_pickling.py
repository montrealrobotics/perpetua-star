import re

from perpetua2.utils.filter_state import Object

from tests.fixed_input_predictions import PROCTHOR_ESTIMATOR


def test_saved_estimator_has_no_absolute_paths(tmp_path):
    path = tmp_path / "object.pkl"
    Object.load(str(PROCTHOR_ESTIMATOR)).dump(str(path))
    assert not re.search(rb"/home/|/Users/|/network/|site-packages", path.read_bytes())
