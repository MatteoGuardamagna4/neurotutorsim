"""Verification gate F: SMD/Mahalanobis n=1 guard, and the duration caliper check."""

import pytest

from src.generation import matching as M


def test_smd_raises_at_n_below_minimum():
    with pytest.raises(ValueError):
        M.standardized_mean_difference([430.0], [420.0])


def test_mahalanobis_raises_at_n_below_minimum():
    with pytest.raises(ValueError):
        M.mahalanobis_distance([430.0, 5.2], [420.0, 5.4], sample_matrix=[[430.0, 5.2]])


def test_smd_computable_with_enough_units():
    values_a = [430, 420, 440, 410, 450, 425, 435, 415, 445, 400]
    values_b = [400, 410, 405, 415, 395, 420, 402, 408, 398, 412]
    result = M.standardized_mean_difference(values_a, values_b)
    assert isinstance(result, float)


def test_duration_caliper_pass():
    result = M.duration_caliper_check(duration_anchor=430.0, duration_other=460.0)
    assert result["passed"] is True
    assert result["relative_difference"] == pytest.approx(30 / 430)


def test_duration_caliper_fail():
    result = M.duration_caliper_check(duration_anchor=430.0, duration_other=600.0)
    assert result["passed"] is False


def test_raw_feature_deltas():
    anchor = {"words": 435, "fk_grade": 8.0, "condition": "traditional"}
    other = {"words": 414, "fk_grade": 7.5, "condition": "ai_scaffolding"}
    deltas = M.raw_feature_deltas(anchor, other)
    assert deltas == {"words": -21, "fk_grade": pytest.approx(-0.5)}
