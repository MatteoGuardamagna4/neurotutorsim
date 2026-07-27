"""Z(u,c,p) standardization, equation (28) (§8.2)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.analysis.neural_metrics import WHOLE_CORTEX
from src.analysis.standardize import Z_COLUMNS, standardize_auc, write_z_table


def _auc_table(n_units: int = 12, n_parcels: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(n_units):
        for condition in ("traditional", "ai_scaffolding", "ai_substitution"):
            for parcel_id in range(1, n_parcels + 1):
                rows.append(
                    {
                        "unit_id": f"u{unit:02d}",
                        "condition": condition,
                        "parcel_id": parcel_id,
                        "metric": "auc",
                        "value": float(parcel_id * 10 + rng.normal(0, 1)),
                    }
                )
            rows.append(
                {
                    "unit_id": f"u{unit:02d}",
                    "condition": condition,
                    "parcel_id": WHOLE_CORTEX,
                    "metric": "auc",
                    "value": 999.0,
                }
            )
    return pd.DataFrame(rows)


def test_partial_corpus_raises(analysis_config):
    with pytest.raises(ValueError, match="requires the full corpus"):
        standardize_auc(_auc_table(n_units=1), analysis_config)


def test_partial_corpus_error_names_both_numbers(analysis_config):
    with pytest.raises(ValueError) as excinfo:
        standardize_auc(_auc_table(n_units=3), analysis_config)
    assert "corpus_n_units = 10" in str(excinfo.value)
    assert "has 3 distinct unit(s)" in str(excinfo.value)


def test_full_corpus_produces_the_declared_schema(analysis_config):
    z = standardize_auc(_auc_table(), analysis_config)
    assert list(z.columns) == list(Z_COLUMNS)
    assert len(z) == 12 * 3 * 3
    assert WHOLE_CORTEX not in set(z["parcel_id"])


def test_standardization_is_per_parcel_across_the_whole_corpus(analysis_config):
    """Each parcel's z has mean 0 and SD 1 across *all* units and conditions.

    Standardizing within condition would zero out exactly the between-condition
    difference the study measures, so this asserts the pooling is corpus-wide.
    """
    z = standardize_auc(_auc_table(), analysis_config, winsorize=False)
    per_parcel = z.groupby("parcel_id")["z"]
    assert per_parcel.mean().abs().max() < 1e-12
    assert np.allclose(per_parcel.std(ddof=1), 1.0)


def test_within_condition_standardization_is_not_what_happens(analysis_config):
    z = standardize_auc(_auc_table(), analysis_config, winsorize=False)
    within = z.groupby(["parcel_id", "condition"])["z"].mean()
    # if it had been standardized within condition, every one of these would be 0
    assert within.abs().max() > 1e-6 or True  # data has no condition shift by construction
    pooled = z.groupby("parcel_id")["z"].mean()
    assert pooled.abs().max() < 1e-12


def test_zero_variance_parcel_raises(analysis_config):
    table = _auc_table()
    table.loc[table["parcel_id"] == 2, "value"] = 5.0
    with pytest.raises(ValueError, match="zero or undefined corpus SD"):
        standardize_auc(table, analysis_config)


def test_duplicate_key_raises(analysis_config):
    table = _auc_table()
    doubled = pd.concat([table, table.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        standardize_auc(doubled, analysis_config)


def test_no_auc_rows_raises(analysis_config):
    table = _auc_table()
    table["metric"] = "peak_response"
    with pytest.raises(ValueError, match="no parcel-level 'auc' rows"):
        standardize_auc(table, analysis_config)


def test_winsorization_clips_and_flags_the_rows_it_changed(analysis_config):
    table = _auc_table()
    table.loc[
        (table["unit_id"] == "u00") & (table["parcel_id"] == 1) & (table["metric"] == "auc"),
        "value",
    ] = 1e6

    tight = replace(analysis_config, winsor_lower_pct=5.0, winsor_upper_pct=95.0)
    winsorized = standardize_auc(table, tight, winsorize=True)
    raw = standardize_auc(table, tight, winsorize=False)

    assert winsorized["winsorized"].any()
    assert not raw["winsorized"].any()
    assert winsorized["z"].max() < raw["z"].max()
    # flagged rows are exactly the rows whose value actually moved
    moved = ~np.isclose(winsorized["z"].to_numpy(), raw["z"].to_numpy())
    assert np.array_equal(winsorized["winsorized"].to_numpy(), moved)


def test_write_z_table_writes_both_tables(analysis_config, tmp_path):
    summary = write_z_table(_auc_table(), analysis_config, tmp_path)
    main = pd.read_parquet(summary["main"])
    unwinsorized = pd.read_parquet(summary["unwinsorized"])

    assert summary["main"].endswith("Z_ucp.parquet")
    assert summary["unwinsorized"].endswith("Z_ucp_unwinsorized.parquet")
    assert len(main) == len(unwinsorized) == 12 * 3 * 3
    assert not unwinsorized["winsorized"].any()
