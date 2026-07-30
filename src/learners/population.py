"""Synthetic learner population: eq. (15) state and eq. (16) parameters (§7.1-7.2).

No participants exist. Every learner here is a draw from a declared generative
model, and every number this module produces is a *simulated* learner property.

The initial state
-----------------
Equation (15) gives each learner a five-component state, every component in
[0, 1]:

======  ====================================================================
`K`     knowledge -- expressed competence, the state eq. (21) grows
`M`     memory strength -- the consolidated trace of eq. (22), which decays
        more slowly than `K` and enters eq. (17) as `kappa * M`
`R`     independent reasoning -- eq. (23); enters eq. (17) as `rho * R` and is
        the state that offloading erodes
`C`     calibration -- implemented as 1 - running Brier (§7.6)
`D`     dependence -- eq. (25); the propensity to lean on support
======  ====================================================================

The components are correlated, so they are drawn from a **Gaussian copula**: a
multivariate normal with the configured correlation matrix, pushed through the
normal CDF to uniforms, then through the inverse CDF of a normal *truncated to
[0, 1]* per component. Truncation rather than clipping is deliberate -- clipping
a normal to [0, 1] piles probability mass onto the two bounds, which would show
up later as a population of learners with exactly zero dependence.

Three prior-knowledge strata (§7.1) with mixture weights 0.30 / 0.50 / 0.20
shift the `K` and `M` marginal means. Nothing else differs between strata.

Per-learner parameters (eq. 16)
-------------------------------
`alpha_i ~ LogNormal(mu_alpha, sigma_alpha)` -- the knowledge acquisition rate
of eq. (21). LogNormal keeps it strictly positive and right-skewed.

`delta_i ~ Beta(a_delta, b_delta)` -- the forgetting rate of eq. (21), and the
base of the eq. (22) memory decay. Beta keeps it inside [0, 1].

Track B. numpy, pandas, scipy; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, truncnorm

from src.learners.config import (
    STATE_COMPONENTS,
    STATE_LOWER,
    STATE_UPPER,
    LearnerConfig,
    PopulationConfig,
)
from src.learners.seeds import SeedStreams

#: Stream that every draw in this module comes from.
POPULATION_STREAM = "population"

#: Columns of the frame `init_population` returns, in order.
POPULATION_COLUMNS: Tuple[str, ...] = (
    "learner_id",
    "stratum",
    "stratum_index",
    *STATE_COMPONENTS,
    "alpha_i",
    "delta_i",
    "confidence_bias",
    "speed_factor",
    "theta_0",
)


def theta_from_k(knowledge: np.ndarray, cfg: PopulationConfig) -> np.ndarray:
    """Ability on the logit scale, from knowledge.

    Equation: theta = theta_intercept + theta_slope * (K - theta_center).

    Direction: `theta_slope > 0`, so **more knowledge means a higher ability and
    a higher predicted probability of a correct response** in eq. (17).

    This is the only channel by which `K` enters the response model; `M` and `R`
    enter eq. (17) through their own terms, so nothing is double-counted.

    ASSUMPTION: the brief states that ability is on the logit scale but not how
    it relates to `K`. A linear map centred on `theta_center` is the simplest
    monotone choice. See `docs/phase3_assumptions.md`.
    """
    return cfg.theta_intercept + cfg.theta_slope * (np.asarray(knowledge, dtype=np.float64) - cfg.theta_center)


def _copula_uniforms(
    n_learners: int, cfg: PopulationConfig, rng: np.random.Generator
) -> np.ndarray:
    """`(n_learners, 5)` uniforms with the configured rank correlation.

    The correlation matrix is validated (symmetric, unit diagonal, positive
    semi-definite) when `PopulationConfig` is constructed. A Cholesky factor is
    used when one exists and an eigendecomposition otherwise, so a *singular*
    but still valid correlation matrix -- two perfectly correlated components,
    say -- works rather than raising from LAPACK.
    """
    correlation = np.asarray(cfg.correlation, dtype=np.float64)
    try:
        factor = np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        eigenvalues, eigenvectors = np.linalg.eigh(correlation)
        factor = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, STATE_LOWER, None)))
    standard = rng.standard_normal(size=(n_learners, len(STATE_COMPONENTS)))
    return norm.cdf(standard @ factor.T)


def _truncated_marginals(
    uniforms: np.ndarray, means: np.ndarray, sds: np.ndarray
) -> np.ndarray:
    """Inverse CDF of a per-learner normal truncated to [0, 1]."""
    lower = (STATE_LOWER - means) / sds
    upper = (STATE_UPPER - means) / sds
    return truncnorm.ppf(uniforms, a=lower, b=upper, loc=means, scale=sds)


def init_population(n_learners: int, cfg: LearnerConfig, seed_seq: SeedStreams) -> pd.DataFrame:
    """Draw a synthetic learner population (§7.1-7.2, eq. 15-16).

    Parameters
    ----------
    n_learners
        Number of simulated learners. Overrides `cfg.population.n_learners`.
    cfg
        Full learner configuration; only the `population` group is read.
    seed_seq
        Named seed streams. Every draw comes from the `population` stream, so
        the population is unchanged by anything the episodes later do.

    Returns
    -------
    pandas.DataFrame
        `POPULATION_COLUMNS`, one row per learner, indexed by `learner_id`.

    Raises
    ------
    ValueError
        If `n_learners < 1`, or if any drawn state component escapes [0, 1]
        (which would mean the truncation failed, not that a learner is unusual).
    """
    if n_learners < 1:
        raise ValueError(f"n_learners must be >= 1; got {n_learners}")
    population = cfg.population
    rng = seed_seq.generator(POPULATION_STREAM)

    weights = np.asarray(population.stratum_weights, dtype=np.float64)
    stratum_index = rng.choice(population.n_strata, size=n_learners, p=weights)

    means = np.tile(np.asarray(population.state_means, dtype=np.float64), (n_learners, 1))
    sds = np.tile(np.asarray(population.state_sds, dtype=np.float64), (n_learners, 1))
    k_column = STATE_COMPONENTS.index("K")
    m_column = STATE_COMPONENTS.index("M")
    means[:, k_column] += np.asarray(population.stratum_k_shift, dtype=np.float64)[stratum_index]
    means[:, m_column] += np.asarray(population.stratum_m_shift, dtype=np.float64)[stratum_index]

    uniforms = _copula_uniforms(n_learners, population, rng)
    state = _truncated_marginals(uniforms, means, sds)

    if not np.isfinite(state).all():
        raise ValueError(
            "the truncated-normal marginals produced non-finite state values; check "
            "population.state_means / state_sds and the stratum shifts"
        )
    if state.min() < STATE_LOWER or state.max() > STATE_UPPER:
        raise ValueError(
            f"drawn state escaped the eq. (15) bounds [{STATE_LOWER}, {STATE_UPPER}]: "
            f"observed [{state.min()}, {state.max()}]. The marginals are truncated, not "
            f"clipped, so this indicates a broken inverse-CDF call rather than an outlier."
        )

    alpha = rng.lognormal(mean=population.mu_alpha, sigma=population.sigma_alpha, size=n_learners)
    delta = rng.beta(a=population.a_delta, b=population.b_delta, size=n_learners)
    confidence_bias = rng.normal(
        loc=population.confidence_bias_mu, scale=population.confidence_bias_sd, size=n_learners
    )
    speed_factor = rng.lognormal(mean=STATE_LOWER, sigma=population.speed_sigma, size=n_learners)

    frame = pd.DataFrame(
        {
            "learner_id": np.arange(n_learners, dtype=np.int64),
            "stratum": np.asarray(population.stratum_names, dtype=object)[stratum_index],
            "stratum_index": stratum_index.astype(np.int64),
            **{name: state[:, i] for i, name in enumerate(STATE_COMPONENTS)},
            "alpha_i": alpha,
            "delta_i": delta,
            "confidence_bias": confidence_bias,
            "speed_factor": speed_factor,
        }
    )
    frame["theta_0"] = theta_from_k(frame["K"].to_numpy(), population)
    return frame.loc[:, list(POPULATION_COLUMNS)].set_index("learner_id", drop=False)


# ---------------------------------------------------------------------------
# The compute view
# ---------------------------------------------------------------------------


@dataclass
class LearnerState:
    """Mutable array view of the population, carried through the episode loop.

    One 1-D array per quantity, all of length `n_learners`, all aligned on
    position. This is the shape every equation in this package is vectorised
    over: there is no per-learner Python object, and no loop over learners
    anywhere.

    The counters exist because eq. (25) and the §3.4 persistence policies are
    stated in terms of *history*, not of the current state alone.
    """

    learner_id: np.ndarray
    stratum_index: np.ndarray
    K: np.ndarray
    M: np.ndarray
    R: np.ndarray
    C: np.ndarray
    D: np.ndarray
    alpha: np.ndarray
    delta: np.ndarray
    confidence_bias: np.ndarray
    speed_factor: np.ndarray
    #: Running Brier accumulators behind the §7.6 calibration implementation.
    brier_sum: np.ndarray
    brier_count: np.ndarray
    #: Consecutive episodes closed with an unaided success -- drives the §3.4
    #: fading policies and the eq. (25) `SupportFaded` term.
    independent_success_streak: np.ndarray
    episodes_completed: int = 0

    _ARRAY_FIELDS = (
        "learner_id",
        "stratum_index",
        "K",
        "M",
        "R",
        "C",
        "D",
        "alpha",
        "delta",
        "confidence_bias",
        "speed_factor",
        "brier_sum",
        "brier_count",
        "independent_success_streak",
    )

    @property
    def n_learners(self) -> int:
        return int(self.learner_id.size)

    def theta(self, cfg: PopulationConfig) -> np.ndarray:
        """Current ability on the logit scale, from the current `K`."""
        return theta_from_k(self.K, cfg)

    def copy(self) -> "LearnerState":
        """A deep copy. Every array is copied, so a probe cannot mutate the run.

        §7.7 checkpoint assessments are read-only by construction: they operate
        on `state.copy()`, and `tests/test_checkpoints.py` asserts the original
        is untouched.
        """
        values: Dict[str, object] = {
            name: getattr(self, name).copy() for name in self._ARRAY_FIELDS
        }
        values["episodes_completed"] = self.episodes_completed
        return LearnerState(**values)  # type: ignore[arg-type]

    def state_matrix(self) -> np.ndarray:
        """`(n_learners, 5)` view of the eq. (15) state, in `STATE_COMPONENTS` order."""
        return np.column_stack([getattr(self, name) for name in STATE_COMPONENTS])

    def as_frame(self) -> pd.DataFrame:
        """Long-lived columns of the state, for the §4.2 `learner_state` output."""
        return pd.DataFrame(
            {
                "learner_id": self.learner_id,
                **{name: getattr(self, name) for name in STATE_COMPONENTS},
            }
        )


def init_state(population: pd.DataFrame) -> LearnerState:
    """Build the array view from the frame `init_population` returned."""
    missing = [c for c in POPULATION_COLUMNS if c not in population.columns]
    if missing:
        raise ValueError(f"population frame is missing column(s) {missing}")
    n_learners = len(population)
    zeros = np.zeros(n_learners, dtype=np.float64)
    counters = np.zeros(n_learners, dtype=np.int64)
    return LearnerState(
        learner_id=population["learner_id"].to_numpy(dtype=np.int64),
        stratum_index=population["stratum_index"].to_numpy(dtype=np.int64),
        K=population["K"].to_numpy(dtype=np.float64).copy(),
        M=population["M"].to_numpy(dtype=np.float64).copy(),
        R=population["R"].to_numpy(dtype=np.float64).copy(),
        C=population["C"].to_numpy(dtype=np.float64).copy(),
        D=population["D"].to_numpy(dtype=np.float64).copy(),
        alpha=population["alpha_i"].to_numpy(dtype=np.float64),
        delta=population["delta_i"].to_numpy(dtype=np.float64),
        confidence_bias=population["confidence_bias"].to_numpy(dtype=np.float64),
        speed_factor=population["speed_factor"].to_numpy(dtype=np.float64),
        brier_sum=zeros.copy(),
        brier_count=zeros.copy(),
        independent_success_streak=counters.copy(),
    )


def stratum_summary(population: pd.DataFrame) -> pd.DataFrame:
    """Mean initial state per stratum -- the table the §10.2 monotonicity check reads."""
    grouped = population.groupby("stratum_index", as_index=False).agg(
        stratum=("stratum", "first"),
        n_learners=("learner_id", "size"),
        **{name: (name, "mean") for name in STATE_COMPONENTS},
        theta_0=("theta_0", "mean"),
    )
    return grouped.sort_values("stratum_index").reset_index(drop=True)


def _field_names() -> Tuple[str, ...]:  # pragma: no cover - introspection helper
    return tuple(f.name for f in fields(LearnerState))
