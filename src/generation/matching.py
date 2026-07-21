"""Stimulus matching diagnostics (§5.3).

§5.3 equations (2) and (3), standardized mean difference (SMD) and
Mahalanobis distance, both require a pooled covariance / pooled variance
estimated across a *sample* of stimuli. With a single authored unit there is
no sample, and both statistics are mathematically undefined -- not zero, not
approximate. `standardized_mean_difference` and `mahalanobis_distance` raise
ValueError below `MIN_UNITS_FOR_MATCHING`, matching decision gate 16 (§12.1).

What IS computable with a single unit, and what this slice reports instead:
raw per-feature deltas against the traditional anchor, and the §5.5 duration
caliper check (within 10% of the traditional anchor).
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Sequence

MIN_UNITS_FOR_MATCHING = 10


def _require_min_units(*groups: Sequence[Any], context: str) -> None:
    for group in groups:
        if len(group) < MIN_UNITS_FOR_MATCHING:
            raise ValueError(
                f"{context} requires a pooled covariance/variance estimated across a sample "
                f"of stimuli (§5.3), which needs >= {MIN_UNITS_FOR_MATCHING} units; got "
                f"{len(group)}. This is unavailable at n=1 -- not zero, not an approximation."
            )


def standardized_mean_difference(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    """SMD (§5.3 eq. 2) for one feature: (mean_a - mean_b) / pooled_std.

    `values_a`/`values_b` are the per-unit values of one feature within each
    condition. Raises ValueError if either group has fewer than
    MIN_UNITS_FOR_MATCHING units.
    """
    _require_min_units(values_a, values_b, context="standardized_mean_difference")
    mean_a, mean_b = statistics.mean(values_a), statistics.mean(values_b)
    var_a, var_b = statistics.variance(values_a), statistics.variance(values_b)
    pooled_std = math.sqrt((var_a + var_b) / 2)
    if pooled_std == 0:
        raise ValueError("pooled standard deviation is zero; SMD is undefined")
    return (mean_a - mean_b) / pooled_std


def mahalanobis_distance(
    vector_a: Sequence[float], vector_b: Sequence[float], sample_matrix: Sequence[Sequence[float]]
) -> float:
    """Mahalanobis distance (§5.3 eq. 3) between two feature vectors, using
    the covariance estimated from `sample_matrix` (rows = units, columns =
    features). Raises ValueError if the sample has fewer than
    MIN_UNITS_FOR_MATCHING rows.
    """
    _require_min_units(sample_matrix, context="mahalanobis_distance")
    import numpy as np

    sample = np.asarray(sample_matrix, dtype=float)
    cov = np.cov(sample, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    diff = np.asarray(vector_a, dtype=float) - np.asarray(vector_b, dtype=float)
    return float(math.sqrt(diff @ inv_cov @ diff))


def raw_feature_deltas(features_anchor: Dict[str, Any], features_other: Dict[str, Any]) -> Dict[str, float]:
    """Per-feature delta of `features_other` minus `features_anchor`, for the
    numeric fields both share. Computable at any n, including n=1.
    """
    deltas = {}
    for key, anchor_value in features_anchor.items():
        other_value = features_other.get(key)
        if isinstance(anchor_value, (int, float)) and isinstance(other_value, (int, float)):
            deltas[key] = other_value - anchor_value
    return deltas


def duration_caliper_check(
    duration_anchor: float, duration_other: float, tolerance: float = 0.10
) -> Dict[str, Any]:
    """§5.5 caliper check: is `duration_other` within `tolerance` (10% by
    default) of the traditional-condition anchor duration?
    """
    if duration_anchor == 0:
        raise ValueError("anchor duration is zero; caliper check is undefined")
    relative_difference = abs(duration_other - duration_anchor) / duration_anchor
    return {
        "relative_difference": relative_difference,
        "tolerance": tolerance,
        "passed": relative_difference <= tolerance,
    }
