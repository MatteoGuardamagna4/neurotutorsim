"""Transparent logistic response baseline (§7.3, eq. 17-18).

This is the **main engine** of the whole study. Not by preference -- by
arithmetic. Brief §9.3 asks for ~1.2e10 simulated episodes; one LLM call each
would take centuries, so an LLM cannot be the response engine and a transparent
vectorised model is the only thing that fits the budget. Minitaur is a
validation sample (§10.2), never infrastructure.

The two equations
-----------------
Equation (17), unaided::

    P(Y_iut = 1) = logistic(theta_i,t - b_u + rho * R_i,t + kappa * M_i,t)

Equation (18), under support level `h`::

    P(Y_iut^support = 1) = logistic(theta_i,t - b_u + rho * R_i,t + kappa * M_i,t
                                    + omega * h_iut)

Equation (18) *is* equation (17) plus `omega * h`, and at `h = 0` the two are
identical. They are therefore one implementation (`response_logit`) rather than
two that could drift apart. Direction of every term:

============  ===================================================================
`theta_i,t`   ability; **larger raises** P(correct)
`b_u`         unit difficulty; enters negatively, so **larger lowers** P(correct)
`rho * R`     independent reasoning; `rho > 0`, so **larger raises** P(correct)
`kappa * M`   memory strength; `kappa > 0`, so **larger raises** P(correct)
`omega * h`   support level; `omega > 0` is enforced, so **more support raises**
              the supported probability and nothing else
============  ===================================================================

Vectorisation
-------------
Every function here takes and returns arrays over learners. There is **no Python
loop over learners anywhere in this package** -- at 5,000 learners x 120
episodes x 2,000 Monte Carlo draws a per-learner loop is not slow, it is
infeasible, so it is treated as a bug rather than as an optimisation target.

Non-finite probabilities raise. A silent `np.nan_to_num` here would turn a
broken parameterisation into a population of learners who answer at chance.

Track B. numpy and scipy; never torch.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from scipy.special import expit

from src.learners.config import STATE_LOWER, STATE_UPPER, ResponseConfig

#: How many offending indices an error message lists before truncating.
_MAX_REPORTED_INDICES = 5


def logistic(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic (sigmoid) function.

    `scipy.special.expit`, not `1 / (1 + exp(-x))`: the naive form overflows for
    `x < -700` and returns a warning plus a NaN, which the guards below would
    then correctly refuse -- turning a representable probability of ~0 into a
    hard failure.
    """
    return expit(np.asarray(x, dtype=np.float64))


def _require_finite(name: str, values: np.ndarray, context: Mapping[str, np.ndarray]) -> np.ndarray:
    """Raise naming the offending inputs if `values` is not everywhere finite."""
    array = np.asarray(values, dtype=np.float64)
    bad = ~np.isfinite(array)
    if not bad.any():
        return array
    indices = np.flatnonzero(bad)
    shown = indices[:_MAX_REPORTED_INDICES]
    detail = []
    for key, source in context.items():
        candidate = np.asarray(source, dtype=np.float64)
        if candidate.shape == array.shape:
            detail.append(f"{key}={candidate[shown].tolist()}")
        else:
            detail.append(f"{key}={candidate.tolist()}")
    raise ValueError(
        f"{name} is non-finite for {indices.size} of {array.size} learners "
        f"(first indices {shown.tolist()}). Offending inputs at those indices: "
        + "; ".join(detail)
        + ". Probabilities are not repaired with nan_to_num -- a non-finite logit means a "
        "parameter or a state value is wrong, and answering at chance would hide it."
    )


def response_logit(
    theta: np.ndarray,
    b_u: np.ndarray,
    reasoning: np.ndarray,
    memory: np.ndarray,
    support_level: np.ndarray,
    cfg: ResponseConfig,
) -> np.ndarray:
    """The eq. (17)/(18) linear predictor.

    Equation: `theta - b_u + rho*R + kappa*M + omega*h`. With `h = 0` this is
    eq. (17) exactly; with `h > 0` it is eq. (18).

    Direction: increasing in ability `theta`, in independent reasoning `R`
    (`rho > 0`), in memory strength `M` (`kappa > 0`) and in support `h`
    (`omega > 0`); decreasing in unit difficulty `b_u`, which enters negatively.

    Parameters
    ----------
    theta
        Ability on the logit scale, per learner.
    b_u
        Unit difficulty on the logit scale; broadcast against `theta`.
    reasoning, memory
        The `R` and `M` components of the eq. (15) state.
    support_level
        `h` in [0, 1]. Zero for an unaided response.
    cfg
        Response configuration holding `rho`, `kappa`, `omega`.

    Returns
    -------
    numpy.ndarray
        The linear predictor, per learner.
    """
    return (
        np.asarray(theta, dtype=np.float64)
        - np.asarray(b_u, dtype=np.float64)
        + cfg.rho * np.asarray(reasoning, dtype=np.float64)
        + cfg.kappa * np.asarray(memory, dtype=np.float64)
        + cfg.omega * np.asarray(support_level, dtype=np.float64)
    )


def probability_correct(
    theta: np.ndarray,
    b_u: np.ndarray,
    reasoning: np.ndarray,
    memory: np.ndarray,
    support_level: np.ndarray,
    cfg: ResponseConfig,
) -> np.ndarray:
    """P(correct) from eq. (17)/(18), guarded against non-finite values.

    Equation: `logistic(theta - b_u + rho*R + kappa*M + omega*h)`.

    Direction: as `response_logit`, monotonically transformed -- the logistic is
    strictly increasing, so every term keeps its sign.

    Raises rather than repairing a non-finite probability: a `nan_to_num` here
    would turn a broken parameterisation into a population answering at chance.
    """
    context = {
        "theta": theta,
        "b_u": b_u,
        "R": reasoning,
        "M": memory,
        "h": support_level,
    }
    logit = response_logit(theta, b_u, reasoning, memory, support_level, cfg)
    _require_finite("response logit", logit, context)
    probability = logistic(logit)
    return _require_finite("P(correct)", probability, context)


def draw_correct(probability: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli draw of correctness, one per learner."""
    p = np.asarray(probability, dtype=np.float64)
    if p.min() < STATE_LOWER or p.max() > STATE_UPPER:
        raise ValueError(
            f"P(correct) must lie in [{STATE_LOWER}, {STATE_UPPER}]; got "
            f"[{p.min()}, {p.max()}]"
        )
    return rng.random(size=p.shape) < p


def draw_confidence(
    probability: np.ndarray,
    confidence_bias: np.ndarray,
    cfg: ResponseConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Reported confidence in [0, 1].

    Equation: `confidence = clip(p_correct + bias_i + eps, 0, 1)` with
    `eps ~ Normal(0, confidence_noise_sd)` drawn per response and `bias_i` drawn
    once per learner at initialisation.

    Direction: the bias term shifts a learner's reports systematically
    (over- or under-confidence); the noise term makes them inconsistent.

    ASSUMPTION. The brief requires a confidence to score eq. (27) but specifies
    no confidence model. Without the two stochastic terms, confidence would be a
    deterministic function of `p_correct`, the Brier score would carry no
    calibration information beyond accuracy, and `C` -- implemented as
    1 - running Brier (§7.6) -- would be degenerate. See
    `docs/phase3_assumptions.md`.
    """
    p = np.asarray(probability, dtype=np.float64)
    noise = rng.normal(loc=STATE_LOWER, scale=cfg.confidence_noise_sd, size=p.shape)
    confidence = p + np.asarray(confidence_bias, dtype=np.float64) + noise
    return np.clip(confidence, STATE_LOWER, STATE_UPPER)


def support_request_probability(
    theta: np.ndarray,
    b_u: np.ndarray,
    dependence: np.ndarray,
    cfg: ResponseConfig,
) -> np.ndarray:
    """P(the learner asks for help before attempting).

    Equation: `logistic(s0 + s1*D - s2*(theta - b_u))`.

    Direction: `s1 > 0`, so **more dependence means more requesting**; `s2 > 0`
    applies to `theta - b_u`, so **a learner who is able relative to the unit
    requests less**.

    ASSUMPTION. §7.7 requires a dependence outcome -- "probability of requesting
    assistance before an independent attempt" -- so a request process must
    exist, but the brief gives no equation for one.
    """
    logit = (
        cfg.request_intercept
        + cfg.request_dependence_slope * np.asarray(dependence, dtype=np.float64)
        - cfg.request_ability_slope
        * (np.asarray(theta, dtype=np.float64) - np.asarray(b_u, dtype=np.float64))
    )
    context = {"theta": theta, "b_u": b_u, "D": dependence}
    _require_finite("support-request logit", logit, context)
    return _require_finite("P(support requested)", logistic(logit), context)


def draw_support_request(probability: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli draw of a help request, one per learner."""
    p = np.asarray(probability, dtype=np.float64)
    return rng.random(size=p.shape) < p


def latency_proxy(
    speed_factor: np.ndarray,
    hint_depth: np.ndarray,
    attempt_made: np.ndarray,
    cfg: ResponseConfig,
) -> np.ndarray:
    """Seconds the simulated learner spent on the response.

    Equation: `speed_i * (base + per_hint*hint_depth + attempt*attempt_made)`.

    Direction: reading more hints and making an independent attempt both take
    longer; `speed_i` is a per-learner multiplier (LogNormal, median 1), so a
    larger value means a slower learner.

    ASSUMPTION. §4.2 requires a `latency_proxy` column; the brief gives no
    latency model. This is a bookkeeping proxy, not a reaction-time prediction,
    and no claim is made about human response times.
    """
    return np.asarray(speed_factor, dtype=np.float64) * (
        cfg.latency_base_s
        + cfg.latency_per_hint_s * np.asarray(hint_depth, dtype=np.float64)
        + cfg.latency_attempt_s * np.asarray(attempt_made, dtype=np.float64)
    )


def brier_score(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Equation (27): mean squared difference between confidence and outcome.

    Equation: `Brier = (1/J) * sum_j (Confidence_j - Correct_j)^2`.

    Direction: **lower is better** -- 0 is a perfectly calibrated and perfectly
    sharp forecaster, and 1 is one that is confidently wrong every time. This is
    the one metric in the package whose direction is inverted relative to
    "larger is better", which is why §7.6 has `C = 1 - Brier`.
    """
    reported = np.asarray(confidence, dtype=np.float64)
    outcome = np.asarray(correct, dtype=np.float64)
    if reported.shape != outcome.shape:
        raise ValueError(
            f"confidence and correctness must align; got {reported.shape} and {outcome.shape}"
        )
    if reported.size == 0:
        raise ValueError(
            "the Brier score of an empty set of forecasts is undefined; it is not reported as 0"
        )
    return float(np.mean((reported - outcome) ** 2))
