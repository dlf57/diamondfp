import pytest
import numpy as np
from diamondfp.utils.features import generate_quantiles, feature_scaling


def test_generate_quantiles_basic():
    test_data = {
        "AVG": [0.200, 0.250, 0.300, 0.350],
        "HR": [100, 200, 300, 400],
    }

    stat_features = {
        "AVG": [0.25, 0.5, 0.75],
    }
    result = generate_quantiles(test_data, stat_features)

    assert np.allclose(result["AVG"], [0.2375, 0.275, 0.3125])


def test_generate_quantiles_empty_df():
    test_data = {"AVG": [], "HR": []}
    stat_features = {"AVG": [0.5]}
    with pytest.raises(IndexError):
        # numpy.quantile on empty data raises an error
        generate_quantiles(test_data, stat_features)


def test_feature_scaling_minmax():
    test_data = {"AVG": [0.200, 0.300, 0.400]}

    result = feature_scaling(test_data, ["AVG"], method="minmax")
    assert result["AVG"] == (0.200, 0.400)


def test_feature_scaling_zscore():
    test_data = {"HR": [100, 200, 300, 400]}
    result = feature_scaling(test_data, ["HR"], method="zscore")
    values = np.array(test_data["HR"])
    mean_expected = values.mean()
    std_expected = values.std(ddof=0)
    assert pytest.approx(result["HR"][0]) == mean_expected
    assert pytest.approx(result["HR"][1]) == std_expected


def test_feature_scaling_invalid_method():
    test_data = {"OPS": [0.8, 0.9, 1.0]}
    with pytest.raises(ValueError):
        feature_scaling(test_data, ["OPS"], method="invalid")
