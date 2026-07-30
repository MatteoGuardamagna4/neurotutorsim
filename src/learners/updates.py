"""State updates: equations (21)-(25) (§7.6).

Every equation is implemented as the brief writes it, vectorised over learners,
and clipped to the [0, 1] bounds equation (15) fixes.

    (21)  K <- clip(K + alpha_i * E * F * (1 - K) - delta_i * K, 0, 1)
    (22)  M <- clip((1 - delta_M,i) * M + eta_M * Retrieval
                    + eta_C * CorrectAfterError, 0, 1)
    (23)  R <- clip(R + eta_R * E * TransferSuccess - eta_O * Offloading, 0, 1)
    (24)  C  = clip(1 - running_Brier / brier_max, 0, 1)          [see below]
    (25)  D <- clip(D + eta_D * SupportUsed
                    - eta_F * SupportFaded * IndependentSuccess, 0, 1)

Directions, term by term
------------------------
=====================================  ==================================================
`alpha_i * E * F * (1 - K)`            **raises** `K`; the `(1 - K)` headroom makes gains
                                       shrink as knowledge saturates, and the product
                                       `E * F` means effort without effective instruction
                                       and effective instruction without effort both
                                       teach nothing
`- delta_i * K`                        **lowers** `K` every episode; forgetting is
                                       unconditional, not gated on a practice gap
`(1 - delta_M,i) * M`                  geometric decay of the memory trace
`+ eta_M * Retrieval`                  **raises** `M`; retrieval practice consolidates
`+ eta_C * CorrectAfterError`          **raises** `M`; working past one's own error
                                       consolidates more than being right first time
`+ eta_R * E * TransferSuccess`        **raises** `R`; independent reasoning grows only
                                       when effortful work transfers
`- eta_O * Offloading`                 **lowers** `R`; letting the system produce the
                                       solution erodes it
`+ eta_D * SupportUsed`                **raises** `D`; consuming support builds dependence
`- eta_F * SupportFaded * IndSuccess`  **lowers** `D`, and only when support was withdrawn
                                       *and* the learner still succeeded
=====================================  ==================================================

The one deliberate deviation, mandated by the brief
---------------------------------------------------
§7.6, verbatim: *"The calibration equation should be implemented more
transparently in code as one minus a running Brier score, rescaled to [0, 1]."*
So equation (24) is **not** implemented literally. `C` is a *derived* quantity,
recomputed from per-learner running Brier accumulators rather than incremented,
which also removes the undefined `feedback_gain` term eq. (24) carries. Recorded
in `docs/phase3_assumptions.md` with the brief's own sentence as justification.

`delta_M,i` is derived, not drawn
--------------------------------
Equation (22) subscripts its decay by learner, but equation (16) draws only
`alpha_i` and `delta_i`. So `delta_M,i = clip(m_decay_scale * delta_i, 0, 1)`.
With `m_decay_scale < 1` the memory trace outlasts expressed knowledge, which is
the ordering the two states are meant to have. ASSUMPTION -- see
`docs/phase3_assumptions.md`.

Clipping is counted, never hidden
---------------------------------
Every clip reports how many values it moved. Heavy clipping means the parameters
are wrong -- a population pinned at `K = 1` or `D = 1` is not a finding -- so the
counts are accumulated per episode and written into the run log.

Track B. numpy only; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from src.learners.config import (
    STATE_COMPONENTS,
    STATE_LOWER,
    STATE_UPPER,
    PopulationConfig,
    UpdateConfig,
)
from src.learners.effort import InteractionProxies
from src.learners.population import LearnerState


@dataclass
class ClipCounter:
    """How many values each clipped update moved, and how many it saw.

    A ratio, not a count, is what matters: 12 clipped values out of 5,000 is
    saturation at the tail, and 4,000 out of 5,000 is a broken parameterisation.
    Both are visible in the run log because both numbers are kept.
    """

    clipped: Dict[str, int] = field(default_factory=dict)
    total: Dict[str, int] = field(default_factory=dict)

    def record(self, name: str, n_clipped: int, n_total: int) -> None:
        self.clipped[name] = self.clipped.get(name, 0) + int(n_clipped)
        self.total[name] = self.total.get(name, 0) + int(n_total)

    def merge(self, other: "ClipCounter") -> None:
        for name, count in other.clipped.items():
            self.record(name, count, other.total.get(name, 0))

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        """Run-log record: absolute counts plus the share clipped."""
        out: Dict[str, Dict[str, float]] = {}
        for name in sorted(self.total):
            total = self.total[name]
            clipped = self.clipped.get(name, 0)
            out[name] = {
                "clipped": clipped,
                "total": total,
                "share": (clipped / total) if total else STATE_LOWER,
            }
        return out

    def summary_lines(self) -> Tuple[str, ...]:
        """Human-readable one line per clipped quantity."""
        rows = self.as_dict()
        return tuple(
            f"{name}: {values['clipped']} / {values['total']} clipped "
            f"({values['share']:.4%})"
            for name, values in rows.items()
        )


def clip_state(name: str, values: np.ndarray, counter: ClipCounter) -> np.ndarray:
    """Clip to the eq. (15) bounds and record how many values moved."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(
            f"state component {name!r} is non-finite before clipping. Clipping bounds a value; "
            f"it does not repair one, so this is refused rather than turned into 0 or 1."
        )
    clipped = np.clip(array, STATE_LOWER, STATE_UPPER)
    counter.record(name, int(np.count_nonzero(clipped != array)), array.size)
    return clipped


# ---------------------------------------------------------------------------
# One function per equation
# ---------------------------------------------------------------------------


def update_knowledge(
    knowledge: np.ndarray,
    alpha: np.ndarray,
    delta: np.ndarray,
    effort: np.ndarray,
    effectiveness: np.ndarray,
    counter: ClipCounter,
) -> np.ndarray:
    """Equation (21): knowledge.

    Equation: `K <- clip(K + alpha_i * E * F * (1 - K) - delta_i * K, 0, 1)`.

    Direction: increasing in the *product* of effort `E` and effectiveness `F`, so
    either one at zero means no gain; the `(1 - K)` headroom shrinks gains as
    knowledge saturates. Decreasing in the per-learner forgetting rate
    `delta_i`, which applies every episode whether or not the unit was practised.
    """
    current = np.asarray(knowledge, dtype=np.float64)
    gain = (
        np.asarray(alpha, dtype=np.float64)
        * np.asarray(effort, dtype=np.float64)
        * np.asarray(effectiveness, dtype=np.float64)
        * (STATE_UPPER - current)
    )
    loss = np.asarray(delta, dtype=np.float64) * current
    return clip_state("K", current + gain - loss, counter)


def memory_decay_rate(delta: np.ndarray, cfg: UpdateConfig) -> np.ndarray:
    """The per-learner memory decay rate of eq. (22).

    Equation: `delta_M,i = clip(m_decay_scale * delta_i, 0, 1)`.

    Direction: increasing in the learner's knowledge-forgetting rate, so a learner
    who forgets faster also consolidates less durably. `m_decay_scale < 1` makes
    the memory trace outlast expressed knowledge.

    Clipped because `(1 - delta_M,i)` must stay non-negative: a decay rate above
    one would flip the sign of the memory trace rather than erase it.
    """
    return np.clip(
        cfg.m_decay_scale * np.asarray(delta, dtype=np.float64), STATE_LOWER, STATE_UPPER
    )


def update_memory(
    memory: np.ndarray,
    delta: np.ndarray,
    proxies: InteractionProxies,
    cfg: UpdateConfig,
    counter: ClipCounter,
) -> np.ndarray:
    """Equation (22): memory strength.

    Equation: `M <- clip((1 - delta_M,i) * M + eta_M * Retrieval
    + eta_C * CorrectAfterError, 0, 1)`.

    Direction: decaying geometrically at `delta_M,i`; increasing in retrieval
    practice and in recovering from one's own error. Unlike eq. (21) there is no
    headroom term, so `M` is a linear AR(1) whose interior equilibrium is
    `gain / delta_M,i` -- a learner who forgets very slowly saturates at 1.
    """
    current = np.asarray(memory, dtype=np.float64)
    retained = (STATE_UPPER - memory_decay_rate(delta, cfg)) * current
    gain = cfg.eta_M * np.asarray(proxies.retrieval, dtype=np.float64) + cfg.eta_C * np.asarray(
        proxies.correct_after_error, dtype=np.float64
    )
    return clip_state("M", retained + gain, counter)


def update_reasoning(
    reasoning: np.ndarray,
    effort: np.ndarray,
    proxies: InteractionProxies,
    cfg: UpdateConfig,
    counter: ClipCounter,
) -> np.ndarray:
    """Equation (23): independent reasoning.

    Equation: `R <- clip(R + eta_R * E * TransferSuccess - eta_O * Offloading,
    0, 1)`.

    Direction: increasing only where effortful work *transfers* -- the gain is the
    product of `E` and `TransferSuccess`, so effort that does not transfer buys
    nothing here. Decreasing in offloading, so letting the system produce the
    solution erodes reasoning even on episodes the learner "got right".

    ⚠️ Eq. (23) has **neither** a headroom term nor a decay term, so `R` is a
    bounded random walk with no interior equilibrium: whichever term dominates on
    average, `R` eventually pins at 0 or 1 and every later update is clipped. The
    clip counts make that visible. See `docs/phase3_assumptions.md`.
    """
    current = np.asarray(reasoning, dtype=np.float64)
    gain = (
        cfg.eta_R
        * np.asarray(effort, dtype=np.float64)
        * np.asarray(proxies.transfer_success, dtype=np.float64)
    )
    loss = cfg.eta_O * np.asarray(proxies.offloading, dtype=np.float64)
    return clip_state("R", current + gain - loss, counter)


def accumulate_brier(
    brier_sum: np.ndarray,
    brier_count: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Add one episode's forecasts to the per-learner running Brier state.

    `confidence`, `correct` and `mask` are `(n_learners, n_responses)`: an episode
    scores several responses and equation (27) averages over them.

    `mask` exists because not every learner produces every response -- one who
    never attempted made no unaided forecast. The count advances per learner by
    that learner's own number of scored forecasts, so an absent response is
    excluded from the denominator rather than entered as a confident zero.
    """
    reported = np.atleast_2d(np.asarray(confidence, dtype=np.float64))
    outcome = np.atleast_2d(np.asarray(correct, dtype=np.float64))
    scored = np.atleast_2d(np.asarray(mask, dtype=bool))
    if not reported.shape == outcome.shape == scored.shape:
        raise ValueError(
            f"confidence, correctness and mask must align; got {reported.shape}, "
            f"{outcome.shape} and {scored.shape}"
        )
    squared = np.where(scored, (reported - outcome) ** 2, STATE_LOWER)
    return (
        np.asarray(brier_sum, dtype=np.float64) + squared.sum(axis=1),
        np.asarray(brier_count, dtype=np.float64) + scored.sum(axis=1),
    )


def calibration_from_brier(
    brier_sum: np.ndarray,
    brier_count: np.ndarray,
    fallback: np.ndarray,
    cfg: UpdateConfig,
    counter: ClipCounter,
) -> np.ndarray:
    """Equation (24), implemented as §7.6 mandates: `1 - running Brier`.

    Equation: `C = clip(1 - (brier_sum / brier_count) / brier_max, 0, 1)`.

    Direction: the Brier score is a loss -- lower is better -- so **better
    calibration means a higher `C`**, which is the direction every other state
    component already has.

    Learners with no forecasts yet keep `fallback` (their initial `C` from the
    eq. 15 draw) rather than being assigned a calibration they have not
    demonstrated.
    """
    counts = np.asarray(brier_count, dtype=np.float64)
    observed = counts > STATE_LOWER
    mean_brier = np.divide(
        np.asarray(brier_sum, dtype=np.float64),
        counts,
        out=np.zeros_like(counts),
        where=observed,
    )
    calibration = np.where(
        observed,
        STATE_UPPER - mean_brier / cfg.brier_max,
        np.asarray(fallback, dtype=np.float64),
    )
    return clip_state("C", calibration, counter)


def update_dependence(
    dependence: np.ndarray,
    proxies: InteractionProxies,
    cfg: UpdateConfig,
    counter: ClipCounter,
) -> np.ndarray:
    """Equation (25): dependence.

    Equation: `D <- clip(D + eta_D * SupportUsed
    - eta_F * SupportFaded * IndependentSuccess, 0, 1)`.

    Direction: increasing in the support level actually consumed. Decreasing only
    in the *product* of withdrawn support and independent success, so dependence
    falls when a learner succeeds without support that was taken away -- never
    merely because support happened to be absent.

    ⚠️ Same structural caveat as eq. (23): no headroom, no decay, so `D` pins at a
    bound over a long enough horizon. Under a fading policy the loss term
    eventually dominates and `D` sits clipped at 0; under `ai_substitution` the
    gain term dominates and it sits clipped at 1.
    """
    current = np.asarray(dependence, dtype=np.float64)
    gain = cfg.eta_D * np.asarray(proxies.support_used, dtype=np.float64)
    loss = (
        cfg.eta_F
        * np.asarray(proxies.support_faded, dtype=np.float64)
        * np.asarray(proxies.independent_success, dtype=np.float64)
    )
    return clip_state("D", current + gain - loss, counter)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def apply_updates(
    state: LearnerState,
    proxies: InteractionProxies,
    effort: np.ndarray,
    effectiveness: np.ndarray,
    confidence: np.ndarray,
    correct: np.ndarray,
    cfg: UpdateConfig,
    counter: ClipCounter,
    *,
    mask: np.ndarray,
) -> None:
    """Advance every state component by one episode, in place.

    Order matters only for `C`: the running Brier state absorbs this episode's
    forecasts *before* `C` is recomputed, so a checkpoint taken after the episode
    sees the calibration the episode produced. `K`, `M`, `R` and `D` are computed
    from the pre-update state, so none of them can read another's new value.
    """
    knowledge = update_knowledge(
        state.K, state.alpha, state.delta, effort, effectiveness, counter
    )
    memory = update_memory(state.M, state.delta, proxies, cfg, counter)
    reasoning = update_reasoning(state.R, effort, proxies, cfg, counter)
    dependence = update_dependence(state.D, proxies, cfg, counter)
    brier_sum, brier_count = accumulate_brier(
        state.brier_sum, state.brier_count, confidence, correct, mask
    )
    state.brier_sum = brier_sum
    state.brier_count = brier_count
    state.C = calibration_from_brier(brier_sum, brier_count, state.C, cfg, counter)
    state.K = knowledge
    state.M = memory
    state.R = reasoning
    state.D = dependence
    state.independent_success_streak = np.where(
        np.asarray(proxies.independent_success, dtype=bool),
        state.independent_success_streak + 1,
        0,
    ).astype(np.int64)
    state.episodes_completed += 1


def decay_only(
    state: LearnerState, n_episodes: int, update_cfg: UpdateConfig, counter: ClipCounter
) -> LearnerState:
    """Project a **copy** of the state forward through a no-practice interval.

    Equation: `K <- K * (1 - delta_i)^n` and `M <- M * (1 - delta_M,i)^n` over
    `n` episodes, with every learning input of eq. (21)-(22) set to zero.

    Direction: strictly non-increasing in the interval length, so **a longer
    no-practice gap means lower retention**, and a learner who forgets faster
    loses more.

    Applies the decay terms of eq. (21) and eq. (22) `n_episodes` times with
    every learning input at zero, and leaves `R`, `C` and `D` untouched --
    eq. (23) and eq. (25) have no decay term, and `C` is a function of forecasts
    already made.

    ASSUMPTION: §7.7 requires "retention after a configurable no-practice
    interval" but the brief gives no retention equation. Setting the learning
    inputs of eq. (21)-(22) to zero and iterating the remaining terms is the
    literal reading of those equations under no practice. See
    `docs/phase3_assumptions.md`.
    """
    if n_episodes < 0:
        raise ValueError(f"n_episodes must be >= 0; got {n_episodes}")
    projected = state.copy()
    retention_k = np.power(STATE_UPPER - np.clip(state.delta, STATE_LOWER, STATE_UPPER), n_episodes)
    retention_m = np.power(STATE_UPPER - memory_decay_rate(state.delta, update_cfg), n_episodes)
    projected.K = clip_state("K_retention", state.K * retention_k, counter)
    projected.M = clip_state("M_retention", state.M * retention_m, counter)
    return projected


def state_columns(state: LearnerState) -> Dict[str, np.ndarray]:
    """The eq. (15) components as a name -> array mapping, in canonical order."""
    return {name: getattr(state, name) for name in STATE_COMPONENTS}


def initial_theta(state: LearnerState, cfg: PopulationConfig) -> np.ndarray:
    """Current ability, for callers that hold a state but not the config path."""
    return state.theta(cfg)
