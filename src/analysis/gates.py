"""Sample-size gates shared by the Phase II analysis modules (§12.1).

One place for the thresholds so that contrasts, RSA and the orchestrating
script cannot drift apart on what "enough units" means, and so that the error
text naming a decision gate is written once.

The rule these encode: an estimator that is mathematically undefined at the
current sample size raises. It never returns NaN, zero, or a degenerate
placeholder. A loud failure at n=1 is the correct behaviour today -- the corpus
is one unit and decision gate 16 is open.
"""

from __future__ import annotations

#: Decision gate 16: 10 pilot units passing matching + correctness tests before
#: the corpus is expanded. Every clustered inferential procedure needs at least
#: this many independent units to have a resampling distribution at all.
MIN_UNITS_GATE_16 = 10

#: A within-unit fixed-effects model needs at least two units to have a
#: between-unit dimension to absorb.
MIN_UNITS_FIXED_EFFECTS = 2

#: A representational dissimilarity matrix over fewer than three units has an
#: upper triangle of at most one cell -- there is no geometry to correlate.
MIN_UNITS_RDM = 3


def require_gate_16(n_units: int, what: str = "cluster bootstrap") -> None:
    """Raise unless `n_units` clears decision gate 16."""
    if n_units < MIN_UNITS_GATE_16:
        raise ValueError(
            f"{what} requires n_units >= {MIN_UNITS_GATE_16} (decision gate 16); got {n_units}"
        )


def require_fixed_effects_units(n_units: int, what: str = "unit fixed-effects model") -> None:
    """Raise unless there are enough units for a within-unit model."""
    if n_units < MIN_UNITS_FIXED_EFFECTS:
        raise ValueError(
            f"{what} requires n_units >= {MIN_UNITS_FIXED_EFFECTS}; got {n_units}. Unit fixed "
            f"effects are undefined with a single unit -- the unit indicator is collinear "
            f"with the intercept and no between-unit variation exists to absorb."
        )


def require_rdm_units(n_units: int, what: str = "representational dissimilarity matrix") -> None:
    """Raise unless there are enough units for an RDM to carry information."""
    if n_units < MIN_UNITS_RDM:
        raise ValueError(
            f"{what} requires n_units >= {MIN_UNITS_RDM}; got {n_units}. With fewer units the "
            f"upper triangle of the RDM holds at most one value, so no dissimilarity "
            f"structure exists to compare."
        )
