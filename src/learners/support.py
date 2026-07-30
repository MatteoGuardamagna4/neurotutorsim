"""Deterministic tutor policies (§3.4) and the support-persistence modifier.

**The tutor is code, not an LLM.** This is the same decision taken for the
Phase I transcripts and it holds here for the same three reasons: no API calls,
no model weights, and a policy whose behaviour is readable from the source
rather than inferred from samples. A stochastic tutor would also put a second
uncontrolled generator inside the contrast the study is built on.

The four conditions
-------------------
==================  ==============  ==============  ====================
condition           support `h`     answer revealed  learner attempt
==================  ==============  ==============  ====================
`traditional`       fixed 3-level   after the        required
                    prewritten      ladder is
                    ladder          exhausted
`ai_scaffolding`    graduated,      after an         required
                    error-diagnosis attempt or
                    dependent       ladder exhaustion
`ai_substitution`   maximal,        immediately      not required
                    immediate       after the first
                                    error
`static_ai`         fixed,          as traditional   required
                    non-interactive
==================  ==============  ==============  ====================

Each policy is a pure function of `(state, history, unit, cfg)` plus the tutor
RNG, and returns the support for **one step** of the interaction. The episode
loop calls it repeatedly, with `history.n_hints` growing, until the learner
resolves the item or the ladder is exhausted -- which is how "revealed after the
ladder is exhausted" is represented without a per-learner loop.

Why the distinction between conditions lives in `adaptation`, not in depth
-------------------------------------------------------------------------
`traditional` and `ai_scaffolding` walk ladders of the *same* depth, so a
condition contrast is not also a ladder-length contrast (`SupportConfig`
enforces the equal depth). What differs is `adaptation`: a prewritten ladder
cannot diagnose the learner's actual error, while the scaffolding tutor
diagnoses it with probability `diagnosis_accuracy`. `adaptation` feeds
instructional *effectiveness* (eq. 20), not effort (eq. 19), so scaffolding buys
better-targeted instruction at the same effort -- and substitution collapses
effort instead, by requiring no attempt and revealing the answer. That is the
mechanism the study's hypothesis lives in, so it is stated here rather than
emerging from a coincidence of parameters.

Vectorisation: every field of `SupportOutcome` is an array over learners. The
per-element types in §3.4's field table are the *element* types.

Track B. numpy only; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from src.learners.config import (
    AI_SCAFFOLDING,
    AI_SUBSTITUTION,
    STATE_LOWER,
    STATE_UPPER,
    STATIC_AI,
    TRADITIONAL,
    LearnerConfig,
    SupportConfig,
)
from src.learners.curriculum import UnitView

#: Stream the tutor's error diagnosis is drawn from.
TUTOR_STREAM = "tutor"

#: Persistence policies (§3.4), orthogonal to condition.
IMMEDIATE_WITHDRAWAL = "immediate_withdrawal"
GRADUAL_FADING = "gradual_fading"
PERSISTENT = "persistent"


@dataclass(frozen=True)
class AttemptHistory:
    """What the tutor is allowed to observe about the interaction so far.

    Observable only. A tutor policy that read `K`, `theta` or `alpha` would be a
    tutor with access to the learner's latent state, which no instructional
    system has.
    """

    #: Independent attempts the learner has made this episode.
    n_attempts: np.ndarray
    #: Hints already given this episode.
    n_hints: np.ndarray
    #: Was the first unaided response correct?
    unaided_correct: np.ndarray
    #: Has the learner produced a correct response at any point this episode?
    resolved: np.ndarray
    #: Consecutive prior episodes closed with an unaided success (longitudinal).
    independent_success_streak: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            name: np.asarray(getattr(self, name)).shape
            for name in (
                "n_attempts",
                "n_hints",
                "unaided_correct",
                "resolved",
                "independent_success_streak",
            )
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(f"every AttemptHistory array must have the same shape; got {shapes}")

    @property
    def n_learners(self) -> int:
        return int(np.asarray(self.n_attempts).size)


@dataclass(frozen=True)
class SupportOutcome:
    """The tutor's action for one step, per learner.

    ============  ======================================================
    `h`           realised support level in [0, 1], after persistence
    `hint_depth`  how many hints have been given, including this one
    `answer_provided`  was the solution handed over?
    `attempt_made`     did the learner produce an independent attempt?
    `adaptation`  in [0, 1] -- how well the hint targeted the real error
    `mismatch`    in [0, 1] -- difficulty misfit between unit and learner
    `h_nominal`   `h` *before* the persistence modifier, needed by the
                  eq. (25) `SupportFaded` term
    ============  ======================================================
    """

    h: np.ndarray
    hint_depth: np.ndarray
    answer_provided: np.ndarray
    attempt_made: np.ndarray
    adaptation: np.ndarray
    mismatch: np.ndarray
    h_nominal: np.ndarray

    _FIELDS = (
        "h",
        "hint_depth",
        "answer_provided",
        "attempt_made",
        "adaptation",
        "mismatch",
        "h_nominal",
    )

    def __post_init__(self) -> None:
        shapes = {name: np.asarray(getattr(self, name)).shape for name in self._FIELDS}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"every SupportOutcome array must have the same shape; got {shapes}")
        for name in ("h", "adaptation", "mismatch", "h_nominal"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.min() < STATE_LOWER or values.max() > STATE_UPPER:
                raise ValueError(
                    f"SupportOutcome.{name} must lie in [{STATE_LOWER}, {STATE_UPPER}]; got "
                    f"[{values.min()}, {values.max()}]"
                )


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def difficulty_mismatch(theta: np.ndarray, b_u: np.ndarray, cfg: SupportConfig) -> np.ndarray:
    """Difficulty misfit between a unit and a learner, in [0, 1].

    Equation: `mismatch = clip(|b_u - theta| / mismatch_scale, 0, 1)`.

    Direction: `mismatch` enters eq. (20) negatively, so **a larger misfit lowers
    instructional effectiveness**. It is symmetric: a unit far below the learner's
    ability is as poorly matched as one far above, which is the point of §5.3's
    matching.

    ASSUMPTION. §7.4 requires a `Mismatch` input to eq. (20) but defines neither
    its scale nor its sign convention. See `docs/phase3_assumptions.md`.
    """
    gap = np.abs(np.asarray(b_u, dtype=np.float64) - np.asarray(theta, dtype=np.float64))
    return np.clip(gap / cfg.mismatch_scale, STATE_LOWER, STATE_UPPER)


def apply_persistence(
    h_nominal: np.ndarray, history: AttemptHistory, cfg: SupportConfig
) -> np.ndarray:
    """Support-persistence modifier (§3.4), applied as a multiplier on `h`.

    Equation: `h = h_nominal * m(streak)`, where `m` is 1 under `persistent`,
    `fade_base ** streak` under `gradual_fading`, and
    `0 if streak >= withdrawal_success_threshold else 1` under
    `immediate_withdrawal`.

    Orthogonal to condition: any condition can be run under any policy.

    =======================  ==========================================
    `immediate_withdrawal`   `h -> 0` once the independent-success
                             streak reaches `withdrawal_success_threshold`
    `gradual_fading`         `h -> h * fade_base ** streak`
    `persistent`             `h` unchanged
    =======================  ==========================================

    Direction: every policy is non-increasing in the independent-success streak,
    so **support never grows because a learner is succeeding**. `persistent` is
    the flat case, not a counter-example.
    """
    nominal = np.asarray(h_nominal, dtype=np.float64)
    streak = np.asarray(history.independent_success_streak, dtype=np.float64)
    policy = cfg.persistence_policy
    if policy == PERSISTENT:
        return nominal.copy()
    if policy == IMMEDIATE_WITHDRAWAL:
        withdrawn = streak >= cfg.withdrawal_success_threshold
        return np.where(withdrawn, STATE_LOWER, nominal)
    if policy == GRADUAL_FADING:
        return nominal * np.power(cfg.fade_base, streak)
    raise ValueError(  # pragma: no cover - config validates first
        f"unknown persistence policy {policy!r}"
    )


def _ladder_level(
    ladder: Tuple[float, ...], n_hints: np.ndarray, n_learners: int
) -> Tuple[np.ndarray, np.ndarray]:
    """`(h, depth)` for the next rung of a prewritten ladder.

    Depth is `n_hints + 1`, clipped to the ladder length: a learner who has
    already consumed every rung stays on the last one, and `answer_provided`
    (decided by the caller) is what actually changes at that point.
    """
    levels = np.asarray(ladder, dtype=np.float64)
    depth = np.minimum(np.asarray(n_hints, dtype=np.int64) + 1, levels.size)
    h = levels[depth - 1]
    return np.broadcast_to(h, (n_learners,)).astype(np.float64), depth.astype(np.int64)


def _finish(
    h_nominal: np.ndarray,
    depth: np.ndarray,
    answer_provided: np.ndarray,
    attempt_made: np.ndarray,
    adaptation: np.ndarray,
    theta: np.ndarray,
    unit: UnitView,
    history: AttemptHistory,
    cfg: SupportConfig,
) -> SupportOutcome:
    """Apply the persistence modifier and assemble the outcome."""
    n_learners = history.n_learners
    b_u = np.broadcast_to(np.float64(unit.b_u), (n_learners,))
    return SupportOutcome(
        h=apply_persistence(h_nominal, history, cfg),
        hint_depth=np.asarray(depth, dtype=np.int64),
        answer_provided=np.asarray(answer_provided, dtype=bool),
        attempt_made=np.asarray(attempt_made, dtype=bool),
        adaptation=np.broadcast_to(adaptation, (n_learners,)).astype(np.float64),
        mismatch=difficulty_mismatch(theta, b_u, cfg),
        h_nominal=np.asarray(h_nominal, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# One pure function per condition
# ---------------------------------------------------------------------------


def traditional_policy(
    theta: np.ndarray,
    history: AttemptHistory,
    unit: UnitView,
    cfg: SupportConfig,
    *,
    rng: np.random.Generator,
) -> SupportOutcome:
    """Fixed 3-level prewritten hint ladder; answer after the ladder is exhausted.

    A prewritten ladder cannot see the learner's error, so `adaptation` is the
    constant `adaptation_traditional` and `rng` is unused -- the argument is kept
    so every policy has one signature and the registry stays uniform.
    """
    n_learners = history.n_learners
    h_nominal, depth = _ladder_level(cfg.hint_ladder_h, history.n_hints, n_learners)
    exhausted = np.asarray(history.n_hints, dtype=np.int64) >= cfg.max_hint_depth
    return _finish(
        h_nominal=h_nominal,
        depth=depth,
        answer_provided=exhausted,
        attempt_made=np.ones(n_learners, dtype=bool),
        adaptation=np.float64(cfg.adaptation_traditional),
        theta=theta,
        unit=unit,
        history=history,
        cfg=cfg,
    )


def ai_scaffolding_policy(
    theta: np.ndarray,
    history: AttemptHistory,
    unit: UnitView,
    cfg: SupportConfig,
    *,
    rng: np.random.Generator,
) -> SupportOutcome:
    """Graduated, error-diagnosis dependent; answer only after attempt or exhaustion.

    The tutor diagnoses the learner's actual error with probability
    `diagnosis_accuracy`, drawn per learner per step from the `tutor` stream.
    A successful diagnosis raises `adaptation` from
    `adaptation_scaffolding_base` to `base + gain`; it does **not** change `h`,
    so scaffolding differs from `traditional` in how well instruction is
    targeted rather than in how much of the work it does.
    """
    n_learners = history.n_learners
    h_nominal, depth = _ladder_level(cfg.scaffolding_h, history.n_hints, n_learners)
    diagnosed = rng.random(size=n_learners) < cfg.diagnosis_accuracy
    adaptation = cfg.adaptation_scaffolding_base + cfg.adaptation_scaffolding_gain * diagnosed
    exhausted = np.asarray(history.n_hints, dtype=np.int64) >= cfg.max_hint_depth
    return _finish(
        h_nominal=h_nominal,
        depth=depth,
        answer_provided=exhausted,
        attempt_made=np.ones(n_learners, dtype=bool),
        adaptation=adaptation,
        theta=theta,
        unit=unit,
        history=history,
        cfg=cfg,
    )


def ai_substitution_policy(
    theta: np.ndarray,
    history: AttemptHistory,
    unit: UnitView,
    cfg: SupportConfig,
    *,
    rng: np.random.Generator,
) -> SupportOutcome:
    """Maximal support, immediately; the answer is revealed after the first error.

    No independent attempt is required, so `attempt_made` is False and the
    solution is handed over at the first step. `hint_depth` is the full ladder
    depth because the whole worked solution is shown at once -- which is what
    drives `Explanation` to zero in eq. (19) and `Offloading` to one in eq. (23).
    """
    n_learners = history.n_learners
    h_nominal = np.full(n_learners, cfg.substitution_h, dtype=np.float64)
    depth = np.full(n_learners, cfg.max_hint_depth, dtype=np.int64)
    return _finish(
        h_nominal=h_nominal,
        depth=depth,
        answer_provided=np.ones(n_learners, dtype=bool),
        attempt_made=np.zeros(n_learners, dtype=bool),
        adaptation=np.float64(cfg.adaptation_substitution),
        theta=theta,
        unit=unit,
        history=history,
        cfg=cfg,
    )


def static_ai_policy(
    theta: np.ndarray,
    history: AttemptHistory,
    unit: UnitView,
    cfg: SupportConfig,
    *,
    rng: np.random.Generator,
) -> SupportOutcome:
    """Fixed, non-interactive explanation; answer revealed as in `traditional`.

    Non-interactive means there is no ladder to walk: one fixed level, and if the
    learner still fails, the answer follows. The optional fourth arm of §3.2.
    """
    n_learners = history.n_learners
    h_nominal = np.full(n_learners, cfg.static_ai_h, dtype=np.float64)
    depth = np.ones(n_learners, dtype=np.int64)
    exhausted = np.asarray(history.n_hints, dtype=np.int64) >= max_steps(STATIC_AI, cfg)
    return _finish(
        h_nominal=h_nominal,
        depth=depth,
        answer_provided=exhausted,
        attempt_made=np.ones(n_learners, dtype=bool),
        adaptation=np.float64(cfg.adaptation_static_ai),
        theta=theta,
        unit=unit,
        history=history,
        cfg=cfg,
    )


PolicyFunction = Callable[..., SupportOutcome]

#: The registry the episode loop dispatches through. A condition with no policy
#: is a `KeyError` here rather than a default policy applied silently.
POLICIES: Dict[str, PolicyFunction] = {
    TRADITIONAL: traditional_policy,
    AI_SCAFFOLDING: ai_scaffolding_policy,
    AI_SUBSTITUTION: ai_substitution_policy,
    STATIC_AI: static_ai_policy,
}

#: Number of interaction steps each condition offers before the answer is
#: revealed. `ai_substitution` reveals at the first step; `static_ai` is
#: non-interactive, so it has exactly one.
_SINGLE_STEP = 1


def max_steps(condition: str, cfg: SupportConfig) -> int:
    """How many support steps `condition` walks before revealing the answer."""
    if condition in (TRADITIONAL, AI_SCAFFOLDING):
        return cfg.max_hint_depth
    if condition in (AI_SUBSTITUTION, STATIC_AI):
        return _SINGLE_STEP
    raise KeyError(f"no support policy registered for condition {condition!r}")


def requires_attempt(condition: str) -> bool:
    """Does the condition require the learner to attempt before support?

    Only `ai_substitution` does not (§3.2). This is the flag that decides whether
    a learner who requests help gets it *before* producing any independent work,
    which is the behaviour eq. (25)'s `SupportUsed` term accumulates into `D`.
    """
    if condition not in POLICIES:
        raise KeyError(f"no support policy registered for condition {condition!r}")
    return condition != AI_SUBSTITUTION


def get_policy(condition: str) -> PolicyFunction:
    """The policy function for `condition`."""
    if condition not in POLICIES:
        raise KeyError(
            f"no support policy registered for condition {condition!r}; known conditions are "
            f"{sorted(POLICIES)}"
        )
    return POLICIES[condition]


def support_faded(h: np.ndarray, h_nominal: np.ndarray) -> np.ndarray:
    """The eq. (25) `SupportFaded` input, in [0, 1].

    Definition: `clip((h_nominal - h) / h_nominal, 0, 1)`, and `1.0` wherever
    `h_nominal == 0` -- no support offered at all is fully faded support.

    Direction: 1 means the persistence policy withdrew everything it could,
    0 means it withdrew nothing. In eq. (25) it multiplies `IndependentSuccess`,
    so **dependence falls only when support was withdrawn *and* the learner
    succeeded without it**.

    ASSUMPTION. `SupportFaded` appears only inside eq. (25); the brief never
    defines it. It is an observable of the tutor interaction here, not a free
    parameter.
    """
    realised = np.asarray(h, dtype=np.float64)
    nominal = np.asarray(h_nominal, dtype=np.float64)
    offered = nominal > STATE_LOWER
    faded = np.divide(
        nominal - realised,
        nominal,
        out=np.ones_like(nominal),
        where=offered,
    )
    return np.clip(faded, STATE_LOWER, STATE_UPPER)
