import pandas as pd
import pytest

from first_place_repro.submit_single import build_submission


def test_build_submission_preserves_official_order(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": ["b_2", "a_1"], "tvt": [0.0, 0.0]}).to_csv(sample, index=False)
    predictions = pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [1.5, 2.5]})
    result = build_submission(sample, predictions)
    assert result.to_dict("list") == {"id": ["b_2", "a_1"], "tvt": [2.5, 1.5]}


def test_build_submission_rejects_missing_prediction(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": ["a_1", "b_2"], "tvt": [0.0, 0.0]}).to_csv(sample, index=False)
    predictions = pd.DataFrame({"id": ["a_1"], "tvt": [1.5]})
    with pytest.raises(RuntimeError, match="does not cover"):
        build_submission(sample, predictions)
