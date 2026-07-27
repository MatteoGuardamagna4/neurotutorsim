"""Representational similarity analysis over units (§6.7, equation 14).

The question §6.7 asks is geometric, not about amplitude: does scaffolding
*preserve*, *sharpen* or *homogenize* the relational structure between concepts
that traditional instruction produces?

    (14)  RDM[u, u'] = 1 - corr(z_u, z_u')

where z_u is unit u's parcel-pattern vector in a given condition. Two RDMs are
compared by Spearman correlation over the **upper triangle only** -- the
diagonal is a structural zero, and the lower triangle is the same numbers
again; including either inflates the correlation with information that is not
there.

The permutation test shuffles condition labels **within matched units**, never
across them. Units are matched on observable stimulus properties (§5.3); a
permutation that moves a label between units breaks the matching that makes the
contrast interpretable in the first place.

Track B: numpy, pandas, scipy. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from src.analysis.gates import require_gate_16, require_rdm_units

__all__ = [
    "unit_pattern_vectors",
    "build_rdm",
    "compare_rdms",
    "permutation_test",
    "classify_geometry",
    "SpearmanResult",
    "PermResult",
    "GeometryVerdict",
]

#: Ratio thresholds separating the three §6.7 verdicts. A condition whose mean
#: representational distance sits inside [1 - TOL, 1 + TOL] times the reference
#: is called "preserved"; outside, "sharpened" (more distinct) or "homogenized"
#: (less distinct).
GEOMETRY_TOLERANCE = 0.05


def unit_pattern_vectors(
    metrics_df: pd.DataFrame,
    condition: str,
    *,
    metric: str = "auc",
) -> np.ndarray:
    """Parcel-pattern vectors z_uc, one row per unit, in unit_id order.

    Returns an `(n_units, n_parcels)` array. Use `pattern_unit_order` for the
    matching row labels -- the two are always computed the same way, so they
    cannot disagree.

    Whole-cortex rows (`parcel_id < 0`) are excluded: a pattern vector is over
    parcels, and mixing a summary statistic into it would weight that summary as
    if it were one more parcel.
    """
    frame = _pattern_frame(metrics_df, condition, metric)
    wide = frame.pivot_table(
        index="unit_id", columns="parcel_id", values="value", aggfunc="first"
    ).sort_index()
    if wide.isna().to_numpy().any():
        incomplete = wide.index[wide.isna().any(axis=1)].tolist()
        raise ValueError(
            f"unit(s) {incomplete} are missing at least one parcel for condition "
            f"{condition!r}, metric {metric!r}; a pattern vector with a hole cannot be "
            f"correlated against a complete one"
        )
    return wide.to_numpy(dtype=np.float64)


def pattern_unit_order(
    metrics_df: pd.DataFrame, condition: str, *, metric: str = "auc"
) -> List[str]:
    """The unit ids labelling the rows of `unit_pattern_vectors`."""
    frame = _pattern_frame(metrics_df, condition, metric)
    return sorted(frame["unit_id"].unique().tolist())


def _pattern_frame(metrics_df: pd.DataFrame, condition: str, metric: str) -> pd.DataFrame:
    required = {"unit_id", "condition", "parcel_id", "metric", "value"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise ValueError(f"metrics table is missing column(s) {missing}")
    frame = metrics_df[
        (metrics_df["condition"] == condition)
        & (metrics_df["metric"] == metric)
        & (metrics_df["parcel_id"] >= 0)
    ]
    if frame.empty:
        raise ValueError(
            f"no parcel-level rows for condition {condition!r} and metric {metric!r}"
        )
    return frame


def build_rdm(patterns: np.ndarray) -> np.ndarray:
    """Equation (14): the representational dissimilarity matrix, 1 - corr.

    `patterns` is `(n_units, n_parcels)`. Requires n_units >= 3: with two units
    the upper triangle holds a single number, which has no structure to compare
    against another RDM.
    """
    matrix = np.asarray(patterns, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"patterns must be a 2-D (n_units, n_parcels) array; got {matrix.shape}")
    require_rdm_units(matrix.shape[0])
    if matrix.shape[1] < 2:
        raise ValueError(
            f"pattern vectors span {matrix.shape[1]} parcel(s); a correlation across parcels "
            f"needs at least 2"
        )
    constant = np.where(matrix.std(axis=1, ddof=0) == 0)[0]
    if constant.size:
        raise ValueError(
            f"unit row(s) {constant.tolist()} are constant across parcels; their correlation "
            f"with any other pattern is undefined"
        )
    rdm = 1.0 - np.corrcoef(matrix)
    np.fill_diagonal(rdm, 0.0)  # exactly zero, not 1 - 0.9999999999
    return (rdm + rdm.T) / 2.0  # kill float asymmetry so the matrix is exactly symmetric


def upper_triangle(rdm: np.ndarray) -> np.ndarray:
    """The strictly upper triangle of an RDM, flattened. Diagonal excluded."""
    matrix = np.asarray(rdm, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"RDM must be square; got shape {matrix.shape}")
    return matrix[np.triu_indices_from(matrix, k=1)]


@dataclass(frozen=True)
class SpearmanResult:
    rho: float
    p_value: float
    n_pairs: int

    def as_dict(self) -> Dict[str, Any]:
        return {"rho": self.rho, "p_value": self.p_value, "n_pairs": self.n_pairs}


def compare_rdms(rdm_a: np.ndarray, rdm_b: np.ndarray) -> SpearmanResult:
    """Spearman correlation between two RDMs, upper triangle only.

    The returned p-value is the analytic Spearman p-value for `n_pairs`
    *pairs*, which are not independent (they share units). It describes the
    monotone association between the two triangles and is not a test of a
    condition effect -- that is what `permutation_test` is for.
    """
    from scipy.stats import spearmanr

    a, b = upper_triangle(rdm_a), upper_triangle(rdm_b)
    if a.shape != b.shape:
        raise ValueError(
            f"RDMs have different sizes ({a.size} vs {b.size} upper-triangle cells); they must "
            f"describe the same set of units"
        )
    if a.size < 2:
        raise ValueError(
            f"comparing RDMs needs at least 2 upper-triangle cells; got {a.size}. That means "
            f"at least 3 units."
        )
    result = spearmanr(a, b)
    return SpearmanResult(
        rho=float(result.statistic), p_value=float(result.pvalue), n_pairs=int(a.size)
    )


@dataclass(frozen=True)
class PermResult:
    observed: float
    p_value: float
    n_perm: int
    n_units: int
    seed: int
    null_mean: float
    null_sd: float
    conditions: Tuple[str, ...]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "observed": self.observed,
            "p_value": self.p_value,
            "n_perm": self.n_perm,
            "n_units": self.n_units,
            "seed": self.seed,
            "null_mean": self.null_mean,
            "null_sd": self.null_sd,
            "conditions": list(self.conditions),
        }


def permutation_test(
    patterns_by_condition: Mapping[str, np.ndarray],
    n_perm: int,
    seed: int,
) -> PermResult:
    """Test whether condition changes representational geometry.

    `patterns_by_condition` maps each condition to its `(n_units, n_parcels)`
    pattern matrix, with rows in the same unit order across conditions.

    The statistic is the mean absolute difference between the conditions' RDM
    upper triangles. The null is built by **shuffling condition labels within
    each unit** -- unit u's traditional and scaffolding patterns may swap with
    each other, but never with unit u''s. Permuting across units would destroy
    the §5.3 matching and test a hypothesis nobody asked.

    Requires n_units >= 10 (decision gate 16).
    """
    conditions = tuple(sorted(patterns_by_condition))
    if len(conditions) < 2:
        raise ValueError(
            f"permutation test needs >= 2 conditions; got {list(conditions)}"
        )
    if n_perm < 1:
        raise ValueError(f"n_perm must be >= 1; got {n_perm}")

    stack = np.stack([np.asarray(patterns_by_condition[c], dtype=np.float64) for c in conditions])
    if stack.ndim != 3:
        raise ValueError("every condition needs a 2-D (n_units, n_parcels) pattern matrix")
    n_conditions, n_units, _ = stack.shape
    require_gate_16(n_units, what="RSA permutation test")

    observed = _rdm_distance(stack)

    rng = np.random.default_rng(seed)
    null = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        permuted = stack.copy()
        for u in range(n_units):
            order = rng.permutation(n_conditions)
            permuted[:, u, :] = stack[order, u, :]
        null[i] = _rdm_distance(permuted)

    # +1 in numerator and denominator: the observed value is itself one draw
    # from the permutation distribution, so p is never exactly zero.
    p_value = float((np.sum(null >= observed) + 1) / (n_perm + 1))
    return PermResult(
        observed=float(observed),
        p_value=p_value,
        n_perm=int(n_perm),
        n_units=int(n_units),
        seed=int(seed),
        null_mean=float(null.mean()),
        null_sd=float(null.std(ddof=1)) if n_perm > 1 else float("nan"),
        conditions=conditions,
    )


def _rdm_distance(stack: np.ndarray) -> float:
    """Mean absolute difference between condition RDMs, upper triangle only."""
    triangles = [upper_triangle(build_rdm(stack[c])) for c in range(stack.shape[0])]
    diffs = [
        np.abs(triangles[i] - triangles[j]).mean()
        for i in range(len(triangles))
        for j in range(i + 1, len(triangles))
    ]
    return float(np.mean(diffs))


@dataclass(frozen=True)
class GeometryVerdict:
    """Structured answer to §6.7's preserve / sharpen / homogenize question.

    A structured object, not prose: the manuscript's wording is written by a
    person from these numbers, and §15 forbids the causal phrasing that a
    generated sentence would drift into.
    """

    condition: str
    reference_condition: str
    verdict: str  # "preserved" | "sharpened" | "homogenized"
    mean_distance: float
    reference_mean_distance: float
    ratio: float
    tolerance: float
    rdm_similarity: Optional[SpearmanResult] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "condition": self.condition,
            "reference_condition": self.reference_condition,
            "verdict": self.verdict,
            "mean_distance": self.mean_distance,
            "reference_mean_distance": self.reference_mean_distance,
            "ratio": self.ratio,
            "tolerance": self.tolerance,
            "rdm_similarity": (
                None if self.rdm_similarity is None else self.rdm_similarity.as_dict()
            ),
        }


def classify_geometry(
    patterns: np.ndarray,
    reference_patterns: np.ndarray,
    *,
    condition: str,
    reference_condition: str,
    tolerance: float = GEOMETRY_TOLERANCE,
) -> GeometryVerdict:
    """Does `condition` preserve, sharpen or homogenize the reference geometry?

    Compares the mean between-unit representational distance under each
    condition. Above the reference by more than `tolerance` -> "sharpened"
    (concepts more distinct); below by more than `tolerance` -> "homogenized";
    within -> "preserved". The RDM-to-RDM Spearman correlation is reported
    alongside, because a condition can hold the *average* distance constant
    while rearranging which units are close to which.
    """
    rdm = build_rdm(patterns)
    reference_rdm = build_rdm(reference_patterns)
    mean_distance = float(upper_triangle(rdm).mean())
    reference_mean = float(upper_triangle(reference_rdm).mean())
    if reference_mean == 0:
        raise ValueError(
            f"reference condition {reference_condition!r} has zero mean representational "
            f"distance; the ratio is undefined"
        )
    ratio = mean_distance / reference_mean
    if ratio > 1 + tolerance:
        verdict = "sharpened"
    elif ratio < 1 - tolerance:
        verdict = "homogenized"
    else:
        verdict = "preserved"
    return GeometryVerdict(
        condition=condition,
        reference_condition=reference_condition,
        verdict=verdict,
        mean_distance=mean_distance,
        reference_mean_distance=reference_mean,
        ratio=ratio,
        tolerance=tolerance,
        rdm_similarity=compare_rdms(rdm, reference_rdm),
    )
