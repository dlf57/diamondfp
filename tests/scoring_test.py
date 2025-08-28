import pytest
import numpy as np
from diamondfp.scoring import (
    tanimoto,
    jaccard,
    manhattan,
    cosine_sim,
)


def test_tanimoto_similarity():
    v1 = [1, 0, 1, 1, 0]
    v2 = [1, 1, 0, 1, 0]
    expected_score = 2 / 4
    assert np.isclose(tanimoto(v1, v2), expected_score)


def test_jaccard_index():
    v1 = [1, 0, 1, 1, 0]
    v2 = [1, 1, 0, 1, 0]
    expected_score = 2 / 4
    assert np.isclose(jaccard(v1, v2), expected_score)


def test_manhattan_distance():
    v1 = [0.2, 0.5, 0.8]
    v2 = [0.1, 0.4, 0.9]
    expected_distance = abs(0.2 - 0.1) + abs(0.5 - 0.4) + abs(0.8 - 0.9)
    assert np.isclose(manhattan(v1, v2), expected_distance)


def test_cosine_similarity():
    v1 = [1, 0, 1]
    v2 = [1, 1, 0]
    dot_product = 1 * 1 + 0 * 1 + 1 * 0
    norm_v1 = np.sqrt(1**2 + 0**2 + 1**2)
    norm_v2 = np.sqrt(1**2 + 1**2 + 0**2)
    expected_score = dot_product / (norm_v1 * norm_v2)
    assert np.isclose(cosine_sim(v1, v2), expected_score)
