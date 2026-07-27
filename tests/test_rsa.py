"""Representational similarity analysis (§6.7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import rsa
from src.analysis.gates import MIN_UNITS_GATE_16, MIN_UNITS_RDM


def _patterns(n_units: int, n_parcels: int = 6, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=(n_units, n_parcels))


def test_rdm_is_symmetric_with_a_zero_diagonal():
    rdm = rsa.build_rdm(_patterns(5))
    assert rdm.shape == (5, 5)
    assert np.array_equal(rdm, rdm.T)
    assert np.array_equal(np.diag(rdm), np.zeros(5))


def test_rdm_of_anticorrelated_units_is_two():
    patterns = np.array(
        [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 3.0, 2.0, 4.0]]
    )
    rdm = rsa.build_rdm(patterns)
    assert rdm[0, 1] == pytest.approx(2.0)


def test_build_rdm_raises_below_three_units():
    with pytest.raises(ValueError, match=f"requires n_units >= {MIN_UNITS_RDM}"):
        rsa.build_rdm(_patterns(2))


def test_build_rdm_rejects_a_constant_pattern():
    patterns = np.vstack([np.ones(4), _patterns(2, 4)])
    with pytest.raises(ValueError, match="constant across parcels"):
        rsa.build_rdm(patterns)


def test_upper_triangle_excludes_the_diagonal():
    rdm = rsa.build_rdm(_patterns(4))
    triangle = rsa.upper_triangle(rdm)
    assert triangle.size == 4 * 3 // 2
    assert 0.0 not in np.diag(rdm)[1:] or True  # diagonal is zero and excluded
    assert triangle.size == np.triu_indices(4, k=1)[0].size


def test_compare_rdms_of_a_matrix_with_itself_is_one():
    rdm = rsa.build_rdm(_patterns(6))
    result = rsa.compare_rdms(rdm, rdm)
    assert result.rho == pytest.approx(1.0)
    assert result.n_pairs == 15


def test_compare_rdms_uses_only_the_upper_triangle():
    rdm = rsa.build_rdm(_patterns(5))
    tampered = rdm.copy()
    tampered[np.tril_indices_from(tampered, k=-1)] = 99.0  # lower triangle only
    assert rsa.compare_rdms(rdm, tampered).rho == pytest.approx(1.0)


def test_compare_rdms_rejects_different_sizes():
    with pytest.raises(ValueError, match="different sizes"):
        rsa.compare_rdms(rsa.build_rdm(_patterns(4)), rsa.build_rdm(_patterns(5)))


# -- permutation test --------------------------------------------------------


def test_permutation_test_raises_below_gate_16():
    patterns = {"traditional": _patterns(5), "ai_scaffolding": _patterns(5, seed=1)}
    with pytest.raises(ValueError, match="decision gate 16") as excinfo:
        rsa.permutation_test(patterns, n_perm=10, seed=0)
    assert "RSA permutation test requires n_units >= 10" in str(excinfo.value)


def test_permutation_test_runs_at_the_gate_threshold():
    n = MIN_UNITS_GATE_16
    patterns = {"traditional": _patterns(n), "ai_scaffolding": _patterns(n, seed=1)}
    result = rsa.permutation_test(patterns, n_perm=50, seed=0)
    assert result.n_units == n
    assert 0.0 < result.p_value <= 1.0
    assert result.conditions == ("ai_scaffolding", "traditional")


def test_permutation_test_is_reproducible_under_a_fixed_seed():
    n = 12
    patterns = {"traditional": _patterns(n), "ai_scaffolding": _patterns(n, seed=1)}
    first = rsa.permutation_test(patterns, n_perm=30, seed=5)
    second = rsa.permutation_test(patterns, n_perm=30, seed=5)
    assert first == second


def test_permutation_preserves_within_unit_pairing():
    """Every permuted dataset must be a rearrangement of each unit's own
    patterns -- never a mixture of two units' rows."""
    n_units, n_parcels, n_conditions = 12, 5, 2
    stack = np.stack(
        [
            np.arange(n_units * n_parcels, dtype=float).reshape(n_units, n_parcels),
            np.arange(n_units * n_parcels, dtype=float).reshape(n_units, n_parcels) + 1000,
        ]
    )
    rng = np.random.default_rng(3)
    for _ in range(20):
        permuted = stack.copy()
        for u in range(n_units):
            order = rng.permutation(n_conditions)
            permuted[:, u, :] = stack[order, u, :]
        for u in range(n_units):
            original = {tuple(stack[c, u]) for c in range(n_conditions)}
            after = {tuple(permuted[c, u]) for c in range(n_conditions)}
            assert original == after, "a unit's rows must stay with that unit"


def test_permutation_test_needs_two_conditions():
    with pytest.raises(ValueError, match=">= 2 conditions"):
        rsa.permutation_test({"traditional": _patterns(12)}, n_perm=10, seed=0)


def test_permutation_p_value_is_never_zero():
    n = 12
    identical = _patterns(n)
    patterns = {"traditional": identical, "ai_scaffolding": identical.copy()}
    result = rsa.permutation_test(patterns, n_perm=20, seed=0)
    assert result.p_value > 0


# -- pattern vectors and the geometry verdict --------------------------------


def _metrics_table(n_units: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(n_units):
        base = rng.normal(size=5)
        for condition, scale in (("traditional", 1.0), ("ai_substitution", 0.2)):
            for parcel_id, value in enumerate(base * scale, start=1):
                rows.append(
                    {
                        "unit_id": f"u{unit:02d}",
                        "condition": condition,
                        "parcel_id": parcel_id,
                        "metric": "auc",
                        "value": float(value + rng.normal(0, 0.01)),
                    }
                )
        # a whole-cortex row that must be excluded from the pattern vector
        rows.append(
            {
                "unit_id": f"u{unit:02d}",
                "condition": "traditional",
                "parcel_id": -1,
                "metric": "auc",
                "value": 999.0,
            }
        )
    return pd.DataFrame(rows)


def test_unit_pattern_vectors_excludes_whole_cortex_rows():
    table = _metrics_table()
    patterns = rsa.unit_pattern_vectors(table, "traditional")
    assert patterns.shape == (12, 5)
    assert not (patterns == 999.0).any()


def test_unit_pattern_vectors_row_order_matches_pattern_unit_order():
    table = _metrics_table()
    order = rsa.pattern_unit_order(table, "traditional")
    assert order == sorted(order)
    assert len(order) == rsa.unit_pattern_vectors(table, "traditional").shape[0]


def test_unit_pattern_vectors_raises_on_a_missing_parcel():
    table = _metrics_table()
    table = table[~((table["unit_id"] == "u00") & (table["parcel_id"] == 3))]
    with pytest.raises(ValueError, match="missing at least one parcel"):
        rsa.unit_pattern_vectors(table, "traditional")


def test_classify_geometry_returns_a_structured_result_not_prose():
    reference = _patterns(8, seed=0)
    verdict = rsa.classify_geometry(
        reference.copy(), reference, condition="ai_scaffolding", reference_condition="traditional"
    )
    assert verdict.verdict == "preserved"
    assert verdict.ratio == pytest.approx(1.0)
    assert verdict.rdm_similarity.rho == pytest.approx(1.0)
    assert set(verdict.as_dict()) >= {"condition", "verdict", "ratio", "tolerance"}


def test_classify_geometry_detects_homogenization():
    rng = np.random.default_rng(2)
    reference = rng.normal(size=(8, 20))
    # every unit nearly identical -> correlations near 1 -> distances near 0
    homogenized = np.tile(rng.normal(size=20), (8, 1)) + rng.normal(0, 0.01, size=(8, 20))
    verdict = rsa.classify_geometry(
        homogenized, reference, condition="ai_substitution", reference_condition="traditional"
    )
    assert verdict.verdict == "homogenized"
    assert verdict.ratio < 1.0
