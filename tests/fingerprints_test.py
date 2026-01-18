import pytest
import pandas as pd
import numpy as np
from diamondfp.fingerprints import binaryfp, binnedfp, normalizedfp, percentilefp, archetypefp


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


def test_percentilefp():
    # Setup simple mock distribution: integers 0 to 99
    distros = {
        "stat1": list(range(100)),
        "stat2": list(range(100))
    }
    
    row = {"stat1": 50, "stat2": 99.5} # 99.5 is > 99, so rank should be 1.0
    
    fp = percentilefp(row, distros)
    
    assert len(fp) == 2
    assert fp[0] == 0.5  # 50 items < 50 out of 100
    assert fp[1] == 1.0  # all 100 items < 99.5


def test_archetypefp_dict():
    # Setup archetypes as dict
    archetypes = {
        "Low": {"stat1": 0, "stat2": 0},
        "High": {"stat1": 10, "stat2": 10}
    }
    
    row = {"stat1": 0, "stat2": 0}
    fp = archetypefp(row, archetypes)
    
    assert len(fp) == 2
    assert fp[0] == 0.0 # Distance to Low
    assert np.isclose(fp[1], np.sqrt(200)) # Distance to High (10^2 + 10^2)


def test_archetypefp_df():
    # Setup archetypes as DataFrame
    df_data = {
        "stat1": [0, 10],
        "stat2": [0, 10]
    }
    archetypes = pd.DataFrame(df_data, index=["Low", "High"])
    
    row = {"stat1": 0, "stat2": 0}
    fp = archetypefp(row, archetypes)
    
    assert len(fp) == 2
    assert fp[0] == 0.0
    assert np.isclose(fp[1], np.sqrt(200))
