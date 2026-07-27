"""Standardized cortical response Z(u,c,p) -- the Phase II -> Phase IV handoff (§8.2).

This module produces the **only** thing Track A hands to Track B: a small
static lookup table indexed by unit, condition and parcel. It enters the
plasticity update as a multiplicand,

    N <- (1 - delta) N + eta * E(i,u,t) * Z(u,c,p)

and nowhere else. Nothing downstream re-reads a TRIBE prediction.

    (28)  z_ucp = (AUC_ucp - mu_p) / sigma_p

with mu_p and sigma_p taken **across the full stimulus corpus** -- every unit,
every condition -- not within condition. Standardizing within condition would
subtract exactly the between-condition difference the study is measuring, and
would do it silently.

Because the moments are corpus-wide, a partial corpus produces a table that
looks fine and is wrong. `standardize_auc` therefore refuses to run until the
corpus is complete, rather than emitting numbers that will be quietly rescaled
by every stimulus authored later.

Track B: numpy, pandas. No torch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from src.analysis.neural_metrics import WHOLE_CORTEX
from src.tribe.config import AnalysisConfig

Z_COLUMNS = ("unit_id", "condition", "parcel_id", "auc", "z", "winsorized")

MAIN_OUTPUT = "Z_ucp.parquet"
UNWINSORIZED_OUTPUT = "Z_ucp_unwinsorized.parquet"


def standardize_auc(
    metrics_df: pd.DataFrame,
    config: AnalysisConfig,
    *,
    winsorize: bool = True,
) -> pd.DataFrame:
    """Equation (28), standardized per parcel across the full corpus.

    Input is the long metrics table (`unit_id`, `condition`, `parcel_id`,
    `metric`, `value`) containing at least the `auc` metric. Whole-cortex rows
    (`parcel_id == WHOLE_CORTEX`) are dropped: Z(u,c,p) is defined per parcel.

    Returns columns `unit_id, condition, parcel_id, auc, z, winsorized`.
    `winsorized` is a per-row boolean recording whether that row's z was
    clipped, so the main table always says which of its own values were
    modified.

    Raises if the corpus has fewer than `config.corpus_n_units` distinct units.
    """
    required = {"unit_id", "condition", "parcel_id", "metric", "value"}
    missing = sorted(required - set(metrics_df.columns))
    if missing:
        raise ValueError(f"metrics table is missing column(s) {missing}")

    frame = metrics_df[
        (metrics_df["metric"] == "auc") & (metrics_df["parcel_id"] != WHOLE_CORTEX)
    ].copy()
    if frame.empty:
        raise ValueError(
            "metrics table contains no parcel-level 'auc' rows; equation (28) standardizes "
            "AUC and has nothing to work on"
        )

    n_units = int(frame["unit_id"].nunique())
    if n_units < config.corpus_n_units:
        raise ValueError(
            f"standardize_auc requires the full corpus: config declares "
            f"corpus_n_units = {config.corpus_n_units} but the metrics table has {n_units} "
            f"distinct unit(s). Equation (28) standardizes against corpus-wide moments, so a "
            f"partial corpus yields a Z table that is silently wrong -- every value would be "
            f"rescaled by units authored later. Complete the corpus, or lower "
            f"analysis.corpus_n_units deliberately and record why."
        )

    duplicated = frame.duplicated(["unit_id", "condition", "parcel_id"])
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} duplicate (unit, condition, parcel) row(s) in the AUC "
            f"table; Z(u,c,p) must have exactly one value per key"
        )

    frame = frame.rename(columns={"value": "auc"})[
        ["unit_id", "condition", "parcel_id", "auc"]
    ].sort_values(["unit_id", "condition", "parcel_id"]).reset_index(drop=True)

    # Corpus-wide moments per parcel: all units, all conditions pooled.
    grouped = frame.groupby("parcel_id")["auc"]
    mu = grouped.transform("mean")
    sigma = grouped.transform(lambda s: s.std(ddof=1))

    degenerate = frame.loc[(sigma.isna()) | (sigma == 0), "parcel_id"].unique()
    if degenerate.size:
        raise ValueError(
            f"parcel(s) {sorted(degenerate.tolist())} have zero or undefined corpus SD of AUC; "
            f"equation (28) divides by sigma_p and is undefined there. This usually means the "
            f"parcel has an identical value in every stimulus."
        )

    frame["z"] = (frame["auc"] - mu) / sigma

    if winsorize:
        lower, upper = _winsor_bounds(frame["z"], config)
        clipped = frame["z"].clip(lower=lower, upper=upper)
        frame["winsorized"] = clipped.ne(frame["z"])
        frame["z"] = clipped
        frame.attrs["winsor_bounds"] = (float(lower), float(upper))
    else:
        frame["winsorized"] = False

    frame.attrs["standardization"] = "per parcel, across the full corpus (all units, all conditions)"
    frame.attrs["corpus_n_units"] = n_units
    frame.attrs["winsorized"] = winsorize
    return frame[list(Z_COLUMNS)]


def _winsor_bounds(z: pd.Series, config: AnalysisConfig) -> Tuple[float, float]:
    """Winsorization limits, computed on the pooled z distribution."""
    return (
        float(np.percentile(z, config.winsor_lower_pct)),
        float(np.percentile(z, config.winsor_upper_pct)),
    )


def write_z_table(
    metrics_df: pd.DataFrame,
    config: AnalysisConfig,
    out_dir: str | Path,
) -> dict:
    """Write both the winsorized main table and the unwinsorized robustness table.

    §8.2 requires both. Writing only the winsorized one would make the
    sensitivity of every downstream result to that clipping unrecoverable
    without a rerun.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)

    main = standardize_auc(metrics_df, config, winsorize=True)
    unwinsorized = standardize_auc(metrics_df, config, winsorize=False)

    main_path = directory / MAIN_OUTPUT
    unwinsorized_path = directory / UNWINSORIZED_OUTPUT
    main.to_parquet(main_path, index=False)
    unwinsorized.to_parquet(unwinsorized_path, index=False)

    return {
        "main": str(main_path),
        "unwinsorized": str(unwinsorized_path),
        "n_rows": int(len(main)),
        "n_winsorized": int(main["winsorized"].sum()),
        "winsor_pct": [config.winsor_lower_pct, config.winsor_upper_pct],
    }
