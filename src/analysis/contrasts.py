"""Within-unit condition contrasts (§6.6, equations 10-13).

Three paired deltas per unit and metric:

    (10)  S - T   ai_scaffolding  minus traditional
    (11)  U - T   ai_substitution minus traditional
    (12)  S - U   ai_scaffolding  minus ai_substitution

Uncertainty comes from a **cluster bootstrap over units and regenerations**
(§6.6). Vertex-level p-values are explicitly prohibited by the brief: parcels
within a stimulus are not independent observations, and treating them as such
inflates the effective sample size by three orders of magnitude. This module
neither computes nor reports them.

Sample-size guards raise rather than degrade. At the current corpus of one
unit, `fit_unit_fe_model` and `cluster_bootstrap` are both expected to fail --
that is the guard working, not a bug.

Track B: numpy, pandas, statsmodels. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.analysis.gates import require_fixed_effects_units, require_gate_16

TRADITIONAL = "traditional"
AI_SCAFFOLDING = "ai_scaffolding"
AI_SUBSTITUTION = "ai_substitution"
CONDITIONS = (TRADITIONAL, AI_SCAFFOLDING, AI_SUBSTITUTION)

#: (label, minuend, subtrahend) for equations (10), (11), (12).
CONTRASTS = (
    ("S-T", AI_SCAFFOLDING, TRADITIONAL),
    ("U-T", AI_SUBSTITUTION, TRADITIONAL),
    ("S-U", AI_SCAFFOLDING, AI_SUBSTITUTION),
)

DELTA_COLUMNS = ("unit_id", "regeneration", "parcel_id", "metric", "contrast", "delta")

__all__ = [
    "CONTRASTS",
    "ModelResult",
    "paired_deltas",
    "fit_unit_fe_model",
    "cluster_bootstrap",
]


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], what: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{what} is missing column(s) {missing}; observed columns: {sorted(frame.columns)}"
        )


def paired_deltas(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Equations (10)-(12): within-unit condition differences.

    Input is the long metrics table with columns `unit_id`, `condition`,
    `parcel_id`, `metric`, `value`, and optionally `regeneration` (defaulting to
    0 when the corpus has no regenerations yet).

    Valid at any n >= 1: a difference between two matched stimuli of the same
    unit needs no pooled variance and no sample. This is the one §6.6 quantity
    that is defined today.

    A unit that is missing one of the two conditions of a contrast is skipped
    for that contrast and reported in `.attrs["incomplete"]` -- never imputed.
    """
    _require_columns(metrics_df, ("unit_id", "condition", "parcel_id", "metric", "value"),
                     "metrics table")
    frame = metrics_df.copy()
    if "regeneration" not in frame.columns:
        frame["regeneration"] = 0

    unknown = sorted(set(frame["condition"]) - set(CONDITIONS))
    if unknown:
        raise ValueError(
            f"metrics table contains unknown condition(s) {unknown}; expected {list(CONDITIONS)}"
        )

    keys = ["unit_id", "regeneration", "parcel_id", "metric"]
    duplicated = frame.duplicated(keys + ["condition"])
    if duplicated.any():
        raise ValueError(
            f"metrics table has {int(duplicated.sum())} duplicate (unit, regeneration, parcel, "
            f"metric, condition) row(s); a paired delta would be ambiguous"
        )

    wide = frame.pivot_table(
        index=keys, columns="condition", values="value", aggfunc="first"
    ).reset_index()

    rows: List[pd.DataFrame] = []
    incomplete: List[str] = []
    for label, plus, minus in CONTRASTS:
        if plus not in wide.columns or minus not in wide.columns:
            incomplete.append(f"{label}: condition(s) absent from the metrics table")
            continue
        block = wide[keys + [plus, minus]].dropna(subset=[plus, minus])
        if block.empty:
            incomplete.append(f"{label}: no row has both conditions")
            continue
        rows.append(
            pd.DataFrame(
                {
                    "unit_id": block["unit_id"].to_numpy(),
                    "regeneration": block["regeneration"].to_numpy(),
                    "parcel_id": block["parcel_id"].to_numpy(),
                    "metric": block["metric"].to_numpy(),
                    "contrast": label,
                    "delta": (block[plus] - block[minus]).to_numpy(),
                }
            )
        )

    if not rows:
        raise ValueError(
            "no paired delta could be formed: the metrics table has no unit with two "
            f"conditions of any contrast. Incomplete: {incomplete}"
        )
    out = pd.concat(rows, ignore_index=True)[list(DELTA_COLUMNS)]
    out.attrs["incomplete"] = incomplete
    return out


@dataclass(frozen=True)
class ModelResult:
    """Equation (13) fit, reported without vertex-level p-values."""

    metric: str
    n_units: int
    n_observations: int
    formula: str
    params: Dict[str, float]
    bse: Dict[str, float]
    conf_int: Dict[str, tuple]
    covariates: List[str] = field(default_factory=list)
    note: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "n_units": self.n_units,
            "n_observations": self.n_observations,
            "formula": self.formula,
            "params": self.params,
            "bse": self.bse,
            "conf_int": {k: list(v) for k, v in self.conf_int.items()},
            "covariates": self.covariates,
            "note": self.note,
        }


def fit_unit_fe_model(
    deltas_df: pd.DataFrame,
    covariates: Optional[Sequence[str]] = None,
    *,
    metric: Optional[str] = None,
) -> ModelResult:
    """Equation (13): condition indicators with unit fixed effects.

    Consumes the unit x condition **level** table (`unit_id`, `condition`,
    `value`, plus any matching-feature covariates), not the paired-delta table.
    Equation (13) estimates condition indicators while unit fixed effects absorb
    the between-unit variation; a within-unit delta has already differenced that
    variation out, so unit dummies fitted on deltas are collinear with the
    intercept and estimate nothing. Passing a delta table raises with that
    explanation rather than silently fitting a different model.

    Requires n_units >= 2.
    """
    frame = deltas_df.copy()
    if "condition" not in frame.columns and "contrast" in frame.columns:
        raise ValueError(
            "fit_unit_fe_model was given a paired-delta table (it has a 'contrast' column). "
            "Equation (13) is a level model: unit fixed effects absorb between-unit variation, "
            "which within-unit deltas have already removed. Pass the unit x condition metrics "
            "table instead."
        )
    _require_columns(frame, ("unit_id", "condition", "value"), "level table")

    if metric is not None:
        if "metric" not in frame.columns:
            raise ValueError("cannot select a metric: the table has no 'metric' column")
        frame = frame[frame["metric"] == metric]
        if frame.empty:
            raise ValueError(f"no rows for metric {metric!r}")
    elif "metric" in frame.columns and frame["metric"].nunique() > 1:
        raise ValueError(
            f"the table mixes {frame['metric'].nunique()} metrics; pass metric=... to select "
            f"one. Fitting one model across metrics with different units is meaningless."
        )

    n_units = int(frame["unit_id"].nunique())
    require_fixed_effects_units(n_units)

    covariate_list = list(covariates or [])
    missing = [c for c in covariate_list if c not in frame.columns]
    if missing:
        raise ValueError(f"covariate(s) {missing} are not columns of the table")

    import statsmodels.formula.api as smf

    terms = ["C(condition, Treatment(reference='traditional'))", "C(unit_id)"] + covariate_list
    formula = "value ~ " + " + ".join(terms)
    fit = smf.ols(formula, data=frame).fit()

    return ModelResult(
        metric=metric or (frame["metric"].iloc[0] if "metric" in frame.columns else "value"),
        n_units=n_units,
        n_observations=int(len(frame)),
        formula=formula,
        params={k: float(v) for k, v in fit.params.items()},
        bse={k: float(v) for k, v in fit.bse.items()},
        conf_int={k: (float(lo), float(hi)) for k, (lo, hi) in fit.conf_int().iterrows()},
        covariates=covariate_list,
        note=(
            "Standard errors are model-based and assume independent units. Inference on "
            "condition effects is reported from the cluster bootstrap, not from these "
            "standard errors; parcel-level p-values are not computed (§6.6)."
        ),
    )


def cluster_bootstrap(
    deltas_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Percentile bootstrap over **units and regenerations** (§6.6).

    Units are resampled with replacement; within each drawn unit its
    regenerations are resampled with replacement. Parcels are *not* resampled --
    they are repeated measurements inside a cluster, not independent draws, and
    resampling them is what produces the naive vertex-level p-values the brief
    prohibits.

    Requires n_units >= 10 (decision gate 16).

    Returns one row per (contrast, metric, parcel_id) with the point estimate,
    the bootstrap SE, and a percentile interval.
    """
    _require_columns(deltas_df, DELTA_COLUMNS, "deltas table")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1; got {n_boot}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    units = np.array(sorted(deltas_df["unit_id"].unique()))
    require_gate_16(len(units), what="cluster bootstrap")

    rng = np.random.default_rng(seed)
    keys = ["contrast", "metric", "parcel_id"]
    point = deltas_df.groupby(keys, sort=True)["delta"].mean()

    # Pre-split by unit once; the inner loop then only concatenates.
    by_unit = {unit: block for unit, block in deltas_df.groupby("unit_id", sort=True)}

    draws = np.empty((n_boot, len(point)), dtype=np.float64)
    for b in range(n_boot):
        drawn_units = rng.choice(units, size=len(units), replace=True)
        parts = []
        for unit in drawn_units:
            block = by_unit[unit]
            regenerations = block["regeneration"].unique()
            chosen = rng.choice(regenerations, size=len(regenerations), replace=True)
            for regeneration in chosen:
                parts.append(block[block["regeneration"] == regeneration])
        resampled = pd.concat(parts, ignore_index=True)
        draws[b] = (
            resampled.groupby(keys, sort=True)["delta"].mean().reindex(point.index).to_numpy()
        )

    lower = np.nanpercentile(draws, 100 * alpha / 2, axis=0)
    upper = np.nanpercentile(draws, 100 * (1 - alpha / 2), axis=0)

    out = point.reset_index().rename(columns={"delta": "estimate"})
    out["boot_se"] = np.nanstd(draws, axis=0, ddof=1)
    out["ci_lower"] = lower
    out["ci_upper"] = upper
    out["n_boot"] = n_boot
    out["n_units"] = len(units)
    out["seed"] = seed
    out["alpha"] = alpha
    out.attrs["resampling_unit"] = "unit, then regeneration within unit"
    out.attrs["note"] = "parcels are not resampled; no parcel- or vertex-level p-values (§6.6)"
    return out
