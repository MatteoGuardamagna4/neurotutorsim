"""Unit difficulty adapter: a units table -> the logit-scale `b_u` of eq. (17)-(18).

Corpus independence is a design requirement, not a detail. The learner engine
simulates over whatever learning units exist, now and in the future, so nothing
in this package names a unit. The table arrives by explicit path; its unit ids
are opaque strings; and the only thing read out of a unit is its `difficulty`.

The difficulty -> `b_u` map
--------------------------
§5.1 item 5 declares `difficulty` as an integer 1-5. Equations (17)-(18) need a
difficulty on the *logit* scale, and the brief never states what that scale is.
The map is therefore documented, configurable and monotone:

    b_u = b_intercept + b_slope * (difficulty - difficulty_center)

with `b_intercept = 0.0`, `b_slope = 0.6`, `difficulty_center = 3.0` in
`config/learners.yaml`. At the defaults a mid-difficulty unit sits at
`b_u = 0` (neither easier nor harder than the ability scale's origin) and one
difficulty step is 0.6 logits, i.e. roughly a 14-point accuracy swing near
p = 0.5. **This is an ASSUMPTION with no anchor in the brief or in data** -- see
`docs/phase3_assumptions.md` and the supervisor open question in `CLAUDE.md`.

`load_curriculum` refuses a real corpus below decision gate 16. That guard has
no override flag and is not a warning: gate 16 exists precisely because a
plausible-looking run over one unit is the failure mode it was written to stop.

Track B. pandas and numpy only; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.learners.config import CurriculumConfig

#: Columns every units table must carry.
REQUIRED_COLUMNS: Tuple[str, ...] = ("unit_id", "domain", "concept", "difficulty")

#: Columns carried through when present, and only when present. Nothing here is
#: fabricated: a missing column is either defaulted with a logged warning
#: (`coverage`, `correctness` -- see `effort.resolve_effectiveness_inputs`) or
#: simply absent from the output.
OPTIONAL_COLUMNS: Tuple[str, ...] = (
    "provenance",
    "coverage",
    "correctness",
    "prerequisites",
    "reference_answer",
    "transfer_answer",
    "misconception_answer",
    "misconception_description",
)

#: Provenance marking a table produced by `scripts/make_placeholder_units.py`.
#: The gate-16 guard is relaxed for these and *only* these, because a
#: placeholder corpus is a development fixture whose outputs are never results.
SYNTHETIC_PLACEHOLDER = "synthetic_placeholder"

#: Provenance recorded when the source table declares none. Deliberately not
#: `SYNTHETIC_PLACEHOLDER`: an unlabelled table is treated as real, so the
#: gate-16 guard applies to it.
PROVENANCE_UNSPECIFIED = "unspecified"


def difficulty_to_b_u(difficulty: np.ndarray, cfg: CurriculumConfig) -> np.ndarray:
    """Map the integer §5.1 difficulty onto the eq. (17)-(18) logit scale.

    Equation: b_u = b_intercept + b_slope * (difficulty - difficulty_center).

    Direction: `b_u` enters eq. (17) as `theta - b_u`, so **larger `b_u` means a
    harder unit and a lower predicted probability of a correct response**.
    `b_slope > 0` is enforced by `CurriculumConfig`, which is what makes the map
    monotone increasing in difficulty.

    Parameters
    ----------
    difficulty
        Integer difficulties, any shape.
    cfg
        Curriculum configuration holding the map's three parameters.

    Returns
    -------
    numpy.ndarray
        `b_u` on the logit scale, same shape as `difficulty`.
    """
    values = np.asarray(difficulty, dtype=np.float64)
    return cfg.b_intercept + cfg.b_slope * (values - cfg.difficulty_center)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"units table not found: {path}. Pass an explicit --units path; the learner engine "
            f"has no default corpus and does not search for one."
        )
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(
        f"unsupported units table format {path.suffix!r} for {path}; expected .csv or .parquet"
    )


def load_curriculum(path: str | Path, cfg: CurriculumConfig) -> pd.DataFrame:
    """Load a units table and attach the logit-scale unit difficulty `b_u`.

    Parameters
    ----------
    path
        Explicit path to a units table (`.csv` or `.parquet`). There is no
        default: a corpus-independent engine must be told which corpus it is
        simulating over.
    cfg
        Curriculum configuration: the `b_u` map and the gate-16 threshold.

    Returns
    -------
    pandas.DataFrame
        One row per unit, indexed 0..n-1, carrying `REQUIRED_COLUMNS`, any
        present `OPTIONAL_COLUMNS`, a `provenance` column, and `b_u`.

    Raises
    ------
    ValueError
        If a required column is missing, if a difficulty is not an integer in
        the configured range, if unit ids are not unique, or if the table is a
        real corpus below **decision gate 16**.
    """
    table_path = Path(path)
    raw = _read_table(table_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(
            f"{table_path}: units table is missing required column(s) {missing}. "
            f"Required: {list(REQUIRED_COLUMNS)}. Present: {list(raw.columns)}"
        )

    keep = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in raw.columns]
    units = raw.loc[:, keep].copy()

    if units["unit_id"].duplicated().any():
        duplicates = sorted(units.loc[units["unit_id"].duplicated(), "unit_id"].unique())
        raise ValueError(
            f"{table_path}: unit_id must be unique; duplicated {duplicates}. A duplicated unit "
            f"would be simulated twice and counted once by the gate-16 guard."
        )

    difficulty = pd.to_numeric(units["difficulty"], errors="coerce")
    if difficulty.isna().any():
        bad = units.loc[difficulty.isna(), "unit_id"].tolist()
        raise ValueError(
            f"{table_path}: non-numeric difficulty for unit(s) {bad}. §5.1 item 5 declares "
            f"difficulty as an integer {cfg.difficulty_min}-{cfg.difficulty_max}."
        )
    if not np.allclose(difficulty.to_numpy(), np.round(difficulty.to_numpy())):
        bad = units.loc[difficulty != np.round(difficulty), "unit_id"].tolist()
        raise ValueError(
            f"{table_path}: non-integer difficulty for unit(s) {bad}. §5.1 item 5 declares "
            f"difficulty as an integer {cfg.difficulty_min}-{cfg.difficulty_max}; it is not "
            f"rounded here, because a fractional difficulty means the source table is wrong."
        )
    out_of_range = (difficulty < cfg.difficulty_min) | (difficulty > cfg.difficulty_max)
    if out_of_range.any():
        offenders = units.loc[out_of_range, ["unit_id", "difficulty"]].to_dict("records")
        raise ValueError(
            f"{table_path}: difficulty outside the declared range "
            f"[{cfg.difficulty_min}, {cfg.difficulty_max}] for {offenders}"
        )
    units["difficulty"] = difficulty.astype(np.int64)

    if "provenance" not in units.columns:
        units["provenance"] = PROVENANCE_UNSPECIFIED
    units["provenance"] = units["provenance"].fillna(PROVENANCE_UNSPECIFIED).astype(str)

    _require_gate_16(units, table_path, cfg)

    units["b_u"] = difficulty_to_b_u(units["difficulty"].to_numpy(), cfg)
    return units.reset_index(drop=True)


def _require_gate_16(units: pd.DataFrame, table_path: Path, cfg: CurriculumConfig) -> None:
    """Refuse a real corpus below decision gate 16.

    The exemption is narrow by design: it applies only when **every** row is
    stamped `synthetic_placeholder`, because a table that mixes placeholder and
    real units is a real corpus with placeholder padding, and padding a corpus
    to clear a gate is exactly what the gate exists to prevent.
    """
    n_units = len(units)
    if n_units >= cfg.min_units:
        return
    provenance = units["provenance"]
    if (provenance == SYNTHETIC_PLACEHOLDER).all() and n_units > 0:
        return
    stamped = sorted(provenance.unique().tolist())
    raise ValueError(
        f"{table_path}: units table has {n_units} unit(s) but {cfg.min_units} are required "
        f"(decision gate 16). Gate 16 blocks full-corpus processing until 10 units pass the "
        f"§5.4 matching and correctness tests, so no simulation runs over a smaller real "
        f"corpus. Provenance found in this table: {stamped}; the guard is waived only when "
        f"every row is stamped {SYNTHETIC_PLACEHOLDER!r} (see "
        f"scripts/make_placeholder_units.py). There is no override flag."
    )


@dataclass(frozen=True)
class UnitView:
    """One row of the units table, as the episode loop sees it.

    A frozen view rather than a DataFrame row: the episode loop touches one unit
    per episode and per-row pandas indexing inside that loop would dominate the
    runtime of an otherwise vectorised simulation.

    `coverage` and `correctness` are required here, not optional. They are
    resolved -- with a logged warning when the units table lacks the column -- by
    `effort.resolve_effectiveness_columns`, so that the only way to reach a
    `UnitView` is through the function that reports the substitution.
    """

    unit_id: str
    domain: str
    concept: str
    difficulty: int
    b_u: float
    provenance: str
    coverage: float
    correctness: float
    reference_answer: Optional[str] = None
    transfer_answer: Optional[str] = None
    misconception_answer: Optional[str] = None
    misconception_description: Optional[str] = None


#: Columns `units_to_views` requires, beyond `REQUIRED_COLUMNS`.
VIEW_REQUIRED_COLUMNS: Tuple[str, ...] = ("b_u", "provenance", "coverage", "correctness")


def units_to_views(units: pd.DataFrame) -> Tuple[UnitView, ...]:
    """Convert a resolved units frame into immutable per-unit views.

    Raises
    ------
    ValueError
        If `coverage` or `correctness` is absent. They are not defaulted here:
        `effort.resolve_effectiveness_columns` owns that decision and logs it.
    """
    missing = [c for c in VIEW_REQUIRED_COLUMNS if c not in units.columns]
    if missing:
        raise ValueError(
            f"units frame is missing column(s) {missing}. Load it with load_curriculum() and "
            f"pass it through effort.resolve_effectiveness_columns(), which is where a missing "
            f"coverage or correctness column is defaulted *and reported*."
        )

    def _optional(row: pd.Series, name: str) -> Optional[str]:
        if name not in units.columns:
            return None
        value = row[name]
        return None if pd.isna(value) else str(value)

    views = []
    for _, row in units.iterrows():
        views.append(
            UnitView(
                unit_id=str(row["unit_id"]),
                domain=str(row["domain"]),
                concept=str(row["concept"]),
                difficulty=int(row["difficulty"]),
                b_u=float(row["b_u"]),
                provenance=str(row["provenance"]),
                coverage=float(row["coverage"]),
                correctness=float(row["correctness"]),
                reference_answer=_optional(row, "reference_answer"),
                transfer_answer=_optional(row, "transfer_answer"),
                misconception_answer=_optional(row, "misconception_answer"),
                misconception_description=_optional(row, "misconception_description"),
            )
        )
    return tuple(views)


def is_placeholder_corpus(units: pd.DataFrame) -> bool:
    """True when every row is a `synthetic_placeholder`.

    Callers use this to stamp their outputs, so that a table derived from a
    placeholder corpus can never be mistaken for a project result.
    """
    if "provenance" not in units.columns or units.empty:
        return False
    return bool((units["provenance"] == SYNTHETIC_PLACEHOLDER).all())
