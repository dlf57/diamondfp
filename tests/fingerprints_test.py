import pytest
from diamondfp.fingerprints import binaryfp, binnedfp, normalizedfp


def test_binaryfp():
    row = {"AVG": 0.298, "HR": 250}
    feat_quants = {"AVG": [0.250, 0.275, 0.300], "HR": [200, 400, 600]}
    expected_fp = [1, 1, 0, 1, 0, 0]
    assert binaryfp(row, feat_quants) == expected_fp


def test_binnedfp():
    row = {"AVG": 0.298, "HR": 250}
    feat_quants = {"AVG": [0.250, 0.275, 0.300], "HR": [200, 400, 600]}
    expected_fp = [0, 1, 0, 1, 0, 0]
    assert binnedfp(row, feat_quants) == expected_fp


def test_binnedfp_highest_quantile():
    row = {"AVG": 0.305}
    feat_quants = {"AVG": [0.250, 0.275, 0.300]}  # highest quantile is 0.300
    result = binnedfp(row, feat_quants)
    # Only highest quantile matches
    assert result == [0, 0, 1]


def test_binnedfp_lowest_quantile():
    row = {"AVG": 0.260}
    feat_quants = {"AVG": [0.250, 0.275, 0.300]}
    result = binnedfp(row, feat_quants)
    # Only lowest quantile matches
    assert result == [1, 0, 0]


def test_binnedfp_no_match():
    row = {"AVG": 0.200}
    feat_quants = {"AVG": [0.250, 0.275, 0.300]}
    result = binnedfp(row, feat_quants)
    assert result == [0, 0, 0]


def test_normalizedfp_minmax():
    row = {"AVG": 0.298, "HR": 250}
    feat_scaling = {"AVG": (0.200, 0.350), "HR": (0, 700)}
    expected_fp = [(0.298 - 0.200) / (0.350 - 0.200), (250 - 0) / (700 - 0)]
    assert normalizedfp(row, feat_scaling, method="minmax") == expected_fp


def test_normalizedfp_minmax_zero_division():
    row = {"AVG": 0.298, "HR": 250}
    feat_scaling = {"AVG": (0.300, 0.300), "HR": (0, 700)}
    expected_fp = [0.0, (250 - 0) / (700 - 0)]
    assert normalizedfp(row, feat_scaling, method="minmax") == expected_fp


def test_normalizedfp_zscore():
    row = {"AVG": 0.298, "HR": 250}
    feat_scaling = {"AVG": (0.275, 0.025), "HR": (350, 200)}
    expected_fp = [(0.298 - 0.275) / 0.025, (250 - 350) / 200]
    assert normalizedfp(row, feat_scaling, method="zscore") == expected_fp


def test_normalizedfp_zscore_zero_division():
    row = {"AVG": 0.298, "HR": 250}
    feat_scaling = {"AVG": (0.275, 0.0), "HR": (350, 200)}
    expected_fp = [0.0, (250 - 350) / 200]
    assert normalizedfp(row, feat_scaling, method="zscore") == expected_fp


def test_normalizedfp_invalid_method():
    row = {"AVG": 0.298, "HR": 250}
    feat_scaling = {"AVG": (0.200, 0.350), "HR": (0, 700)}
    with pytest.raises(ValueError):
        normalizedfp(row, feat_scaling, method="invalid")
