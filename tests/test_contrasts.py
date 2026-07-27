"""Paired deltas, unit fixed effects and the cluster bootstrap (§6.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import contrasts
from src.analysis.gates import MIN_UNITS_GATE_16


def _metrics(n_units: int, n_parcels: int = 2, metric: str = "auc") -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for unit in range(n_units):
        for condition, shift in (
            ("traditional", 0.0),
            ("ai_scaffolding", 1.0),
            ("ai_substitution", -1.0),
        ):
            for parcel_id in range(1, n_parcels + 1):
                rows.append(
                    {
                        "unit_id": f"u{unit:02d}",
                        "condition": condition,
                        "parcel_id": parcel_id,
                        "metric": metric,
                        "value": float(shift + parcel_id + rng.normal(0, 0.01)),
                    }
                )
    return pd.DataFrame(rows)


# -- paired deltas -----------------------------------------------------------


def test_paired_deltas_are_defined_at_n_equals_one():
    deltas = contrasts.paired_deltas(_metrics(1))
    assert set(deltas["contrast"]) == {"S-T", "U-T", "S-U"}
    assert len(deltas) == 3 * 2  # three contrasts x two parcels


def test_deltas_are_exactly_antisymmetric():
    """Reversing which condition is the minuend must negate the delta exactly.

    `a - b` and `b - a` are exact negations in IEEE arithmetic, so this is
    tested with `==`, not a tolerance.
    """
    frame = _metrics(3)
    swapped = frame.copy()
    swapped["condition"] = frame["condition"].map(
        {
            "traditional": "traditional",
            "ai_scaffolding": "ai_substitution",
            "ai_substitution": "ai_scaffolding",
        }
    )

    key = ["unit_id", "regeneration", "parcel_id", "metric"]
    forward = contrasts.paired_deltas(frame)
    reverse = contrasts.paired_deltas(swapped)
    merged = (
        forward[forward["contrast"] == "S-U"]
        .merge(reverse[reverse["contrast"] == "S-U"], on=key, suffixes=("_f", "_r"))
    )
    assert len(merged) == 3 * 2
    assert np.array_equal(merged["delta_f"], -merged["delta_r"])


def test_the_three_contrasts_satisfy_the_triangle_identity():
    """(S-T) - (U-T) == (S-U) up to floating-point round-off.

    Exact equality does not hold: the left side is two subtractions of
    already-rounded values, the right side is one. The identity is checked to
    machine precision rather than asserted bitwise.
    """
    deltas = contrasts.paired_deltas(_metrics(3))
    wide = deltas.pivot_table(
        index=["unit_id", "regeneration", "parcel_id", "metric"],
        columns="contrast",
        values="delta",
    )
    assert np.allclose(wide["S-T"] - wide["U-T"], wide["S-U"], rtol=0, atol=1e-12)


def test_delta_sign_convention():
    frame = pd.DataFrame(
        {
            "unit_id": ["u0"] * 3,
            "condition": ["traditional", "ai_scaffolding", "ai_substitution"],
            "parcel_id": [1, 1, 1],
            "metric": ["auc"] * 3,
            "value": [10.0, 13.0, 4.0],
        }
    )
    deltas = contrasts.paired_deltas(frame).set_index("contrast")["delta"]
    assert deltas["S-T"] == pytest.approx(3.0)
    assert deltas["U-T"] == pytest.approx(-6.0)
    assert deltas["S-U"] == pytest.approx(9.0)


def test_unknown_condition_raises():
    frame = _metrics(2)
    frame.loc[0, "condition"] = "hybrid"
    with pytest.raises(ValueError, match="unknown condition"):
        contrasts.paired_deltas(frame)


def test_duplicate_rows_raise_rather_than_being_averaged():
    frame = pd.concat([_metrics(2), _metrics(2).head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        contrasts.paired_deltas(frame)


def test_regeneration_defaults_to_zero_when_absent():
    deltas = contrasts.paired_deltas(_metrics(2))
    assert set(deltas["regeneration"]) == {0}


# -- fixed-effects model -----------------------------------------------------


def test_fit_unit_fe_model_raises_at_n_equals_one():
    with pytest.raises(ValueError, match="requires n_units >= 2"):
        contrasts.fit_unit_fe_model(_metrics(1), metric="auc")


def test_fit_unit_fe_model_works_at_n_equals_two():
    result = contrasts.fit_unit_fe_model(_metrics(2), metric="auc")
    assert result.n_units == 2
    assert result.n_observations == 2 * 3 * 2
    assert any("ai_scaffolding" in name for name in result.params)


def test_fit_unit_fe_model_recovers_the_planted_condition_shift():
    result = contrasts.fit_unit_fe_model(_metrics(12), metric="auc")
    scaffolding = next(v for k, v in result.params.items() if "ai_scaffolding" in k)
    substitution = next(v for k, v in result.params.items() if "ai_substitution" in k)
    assert scaffolding == pytest.approx(1.0, abs=0.05)
    assert substitution == pytest.approx(-1.0, abs=0.05)


def test_fit_unit_fe_model_rejects_a_delta_table():
    deltas = contrasts.paired_deltas(_metrics(4))
    with pytest.raises(ValueError, match="paired-delta table"):
        contrasts.fit_unit_fe_model(deltas)


def test_fit_unit_fe_model_refuses_to_mix_metrics():
    mixed = pd.concat([_metrics(3, metric="auc"), _metrics(3, metric="peak_response")])
    with pytest.raises(ValueError, match="mixes 2 metrics"):
        contrasts.fit_unit_fe_model(mixed)


def test_fit_unit_fe_model_rejects_an_unknown_covariate():
    with pytest.raises(ValueError, match="are not columns"):
        contrasts.fit_unit_fe_model(_metrics(3), covariates=["fk_grade"], metric="auc")


# -- cluster bootstrap -------------------------------------------------------


def test_cluster_bootstrap_raises_below_gate_16_with_the_required_message():
    deltas = contrasts.paired_deltas(_metrics(1))
    with pytest.raises(ValueError) as excinfo:
        contrasts.cluster_bootstrap(deltas, n_boot=10, seed=0)
    assert str(excinfo.value) == (
        "cluster bootstrap requires n_units >= 10 (decision gate 16); got 1"
    )


@pytest.mark.parametrize("n_units", [1, 5, MIN_UNITS_GATE_16 - 1])
def test_cluster_bootstrap_raises_for_every_sample_below_the_gate(n_units):
    deltas = contrasts.paired_deltas(_metrics(n_units))
    with pytest.raises(ValueError, match="decision gate 16"):
        contrasts.cluster_bootstrap(deltas, n_boot=10, seed=0)


def test_cluster_bootstrap_runs_at_the_gate_threshold():
    deltas = contrasts.paired_deltas(_metrics(MIN_UNITS_GATE_16))
    result = contrasts.cluster_bootstrap(deltas, n_boot=25, seed=0)
    assert len(result) == 3 * 2  # contrasts x parcels
    assert (result["ci_lower"] <= result["estimate"]).all()
    assert (result["estimate"] <= result["ci_upper"]).all()
    assert set(result["n_units"]) == {MIN_UNITS_GATE_16}


def test_cluster_bootstrap_is_reproducible_under_a_fixed_seed():
    deltas = contrasts.paired_deltas(_metrics(12))
    first = contrasts.cluster_bootstrap(deltas, n_boot=25, seed=42)
    second = contrasts.cluster_bootstrap(deltas, n_boot=25, seed=42)
    pd.testing.assert_frame_equal(first, second)


def test_cluster_bootstrap_differs_under_a_different_seed():
    deltas = contrasts.paired_deltas(_metrics(12))
    first = contrasts.cluster_bootstrap(deltas, n_boot=25, seed=1)
    second = contrasts.cluster_bootstrap(deltas, n_boot=25, seed=2)
    assert not np.allclose(first["boot_se"], second["boot_se"])


def test_cluster_bootstrap_records_its_resampling_unit():
    deltas = contrasts.paired_deltas(_metrics(12))
    result = contrasts.cluster_bootstrap(deltas, n_boot=10, seed=0)
    assert "unit" in result.attrs["resampling_unit"]
    assert "regeneration" in result.attrs["resampling_unit"]
    assert "p-value" not in result.columns
    assert not any("p" == c or c.endswith("_p") for c in result.columns)
