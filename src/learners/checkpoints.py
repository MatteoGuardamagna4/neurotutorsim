"""Checkpoint assessments with all support removed (§7.7).

The §10.6 falsification logic lives here. A learner in the `ai_substitution`
condition performs *well* while support is present -- eq. (18) adds `omega * h`
to their logit, and the answer is often handed over outright. Whether anything
was learned only becomes visible when the support is taken away. So a checkpoint
scores every probe at `h = 0`, except the one deliberate exception: the eq. (26)
support gap, which needs a supported score to subtract an unaided one from.

What is measured
----------------
==============================  =================================================
unaided accuracy (trained)      trained items, `h = 0`
near transfer                   `b_u + near_b_delta`, `h = 0`
far transfer                    `b_u + far_b_delta`, `h = 0`
retention                       trained items after a no-practice interval, on a
                                decay-only projection of the state
Brier score (eq. 27)            over this checkpoint's unaided forecasts
expected calibration error      binned |accuracy - confidence|, same forecasts
support gap (eq. 26)            supported accuracy - unaided accuracy, trained
                                items, `h = probe_h`
dependence                      P(requesting assistance before an independent
                                attempt), averaged over the trained items
==============================  =================================================

**A checkpoint never mutates learner state.** Every probe reads the state and
the retention projection works on `state.copy()`; `tests/test_checkpoints.py`
asserts the caller's state is bit-identical afterwards. A checkpoint that
advanced the state would make the act of measuring change the trajectory.

Probe draws come from the `checkpoints` seed stream, not the `responses` stream,
so adding or moving a checkpoint cannot shift the responses of the episodes
around it.

Track B. numpy and pandas; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd

from src.learners.config import (
    AI_SCAFFOLDING,
    AI_SUBSTITUTION,
    STATE_COMPONENTS,
    STATE_LOWER,
    STATE_UPPER,
    LearnerConfig,
)
from src.learners.curriculum import UnitView
from src.learners.engine import STAGE_PROBE, EpisodeBatch, ResponseEngine
from src.learners.population import LearnerState
from src.learners.responses import support_request_probability
from src.learners.updates import ClipCounter, decay_only

#: `outputs/tables/phase3_checkpoints.csv` columns.
CHECKPOINT_COLUMNS: Tuple[str, ...] = (
    "learner_id",
    "condition",
    "checkpoint_episode",
    "stratum_index",
    "unaided_accuracy_trained",
    "supported_accuracy_trained",
    "support_gap",
    "near_transfer_accuracy",
    "far_transfer_accuracy",
    "retention_accuracy",
    "retention_interval_episodes",
    "brier_score",
    "expected_calibration_error",
    "dependence",
    "n_trained_items",
    "n_near_items",
    "n_far_items",
    *STATE_COMPONENTS,
)


@dataclass(frozen=True)
class ProbeResult:
    """Accuracy and forecasts from one set of probe items."""

    accuracy: np.ndarray
    confidence: np.ndarray
    correct: np.ndarray

    @property
    def n_items(self) -> int:
        return int(np.asarray(self.confidence).shape[1])


def _probe(
    state: LearnerState,
    units: Sequence[UnitView],
    engine: ResponseEngine,
    cfg: LearnerConfig,
    *,
    episode: int,
    condition: str,
    b_delta: float,
    support: float,
) -> ProbeResult:
    """Score every learner on every probe item at a fixed support level.

    Loops over items -- at most a few dozen -- never over learners.
    """
    if not units:
        raise ValueError(
            "a checkpoint probe needs at least one item; an accuracy over zero items is "
            "undefined and is not reported as 0"
        )
    n_learners = state.n_learners
    theta = state.theta(cfg.population)
    support_level = np.full(n_learners, support, dtype=np.float64)
    confidences = []
    corrects = []
    for unit in units:
        batch = EpisodeBatch(
            unit_id=unit.unit_id,
            episode=episode,
            condition=condition,
            stage=STAGE_PROBE,
            b_u=np.full(n_learners, unit.b_u + b_delta, dtype=np.float64),
            theta=theta,
            memory=state.M,
            reasoning=state.R,
            dependence=state.D,
            support_level=support_level,
            confidence_bias=state.confidence_bias,
        )
        outcome = engine.respond(batch)
        confidences.append(np.asarray(outcome.confidence, dtype=np.float64))
        corrects.append(np.asarray(outcome.answer_correct, dtype=bool))
    confidence = np.column_stack(confidences)
    correct = np.column_stack(corrects)
    return ProbeResult(
        accuracy=correct.mean(axis=1), confidence=confidence, correct=correct
    )


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int
) -> np.ndarray:
    """Per-learner expected calibration error over a checkpoint's forecasts.

    Equation: `ECE = sum_b (n_b / J) * |accuracy_b - mean_confidence_b|` over
    `n_bins` equal-width bins of [0, 1].

    Direction: **lower is better** -- 0 means reported confidence matches
    realised accuracy in every bin.

    ASSUMPTION on two counts. §7.7 requires an ECE but names no bin count
    (`checkpoints.ece_bins`, default 10), and it is computed *per learner*
    because §4.2 gives the checkpoint table one row per learner. With a few dozen
    forecasts per learner the estimate is coarse and biased upward; it is
    comparable across conditions, which is what the §10.6 contrast needs, but it
    is not a population calibration estimate. Recorded in
    `docs/phase3_assumptions.md`.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1; got {n_bins}")
    reported = np.atleast_2d(np.asarray(confidence, dtype=np.float64))
    outcome = np.atleast_2d(np.asarray(correct, dtype=np.float64))
    if reported.shape != outcome.shape:
        raise ValueError(
            f"confidence and correctness must align; got {reported.shape} and {outcome.shape}"
        )
    n_forecasts = reported.shape[1]
    if n_forecasts == 0:
        raise ValueError("expected calibration error over zero forecasts is undefined")
    # `min` keeps confidence == 1.0 inside the last bin rather than one past it.
    bins = np.minimum((reported * n_bins).astype(np.int64), n_bins - 1)
    error = np.zeros(reported.shape[0], dtype=np.float64)
    for index in range(n_bins):
        in_bin = bins == index
        count = in_bin.sum(axis=1)
        occupied = count > 0
        if not occupied.any():
            continue
        safe = np.where(occupied, count, 1)
        mean_confidence = np.where(in_bin, reported, STATE_LOWER).sum(axis=1) / safe
        mean_accuracy = np.where(in_bin, outcome, STATE_LOWER).sum(axis=1) / safe
        error += np.where(
            occupied, (count / n_forecasts) * np.abs(mean_accuracy - mean_confidence), STATE_LOWER
        )
    return error


def _select_items(
    units_seen: Sequence[UnitView], n_items: int, rng: np.random.Generator
) -> Tuple[UnitView, ...]:
    """Pick probe items from the units taught so far.

    Sampled without replacement when the corpus allows it and with replacement
    when it does not, so a short corpus produces a shorter *unique* item list
    rather than a silently truncated probe. Every learner sees the same items:
    an assessment with a per-learner form would confound ability with form
    difficulty.
    """
    if not units_seen:
        raise ValueError(
            "no units have been taught yet, so there are no trained items to probe. A checkpoint "
            "at episode 0 is a specification error, not an empty result."
        )
    pool = np.arange(len(units_seen))
    replace = len(pool) < n_items
    chosen = rng.choice(pool, size=n_items, replace=replace)
    return tuple(units_seen[int(index)] for index in chosen)


def run_checkpoint(
    state: LearnerState,
    units_seen: Sequence[UnitView],
    condition: str,
    engine: ResponseEngine,
    cfg: LearnerConfig,
    rng: np.random.Generator,
    *,
    episode: int,
    counter: ClipCounter,
) -> pd.DataFrame:
    """Run one §7.7 checkpoint. Returns one row per learner.

    Parameters
    ----------
    state
        Learner state, **read only**. Not advanced, not modified.
    units_seen
        Units taught so far; trained items are drawn from these.
    condition
        The instructional condition this arm is running.
    engine
        Probe engine, built on the `checkpoints` stream.
    cfg
        Full learner configuration.
    rng
        The `checkpoints` stream generator, used to select probe items.
    episode
        Episode index the checkpoint follows.
    counter
        Clip counter; the retention projection clips and reports.

    Returns
    -------
    pandas.DataFrame
        `CHECKPOINT_COLUMNS`, one row per learner.
    """
    checkpoint_cfg = cfg.checkpoints
    trained = _select_items(units_seen, checkpoint_cfg.n_trained_items, rng)
    near = _select_items(units_seen, checkpoint_cfg.n_near_items, rng)
    far = _select_items(units_seen, checkpoint_cfg.n_far_items, rng)

    unaided = _probe(
        state, trained, engine, cfg,
        episode=episode, condition=condition, b_delta=STATE_LOWER, support=STATE_LOWER,
    )
    supported = _probe(
        state, trained, engine, cfg,
        episode=episode, condition=condition,
        b_delta=STATE_LOWER, support=checkpoint_cfg.probe_h,
    )
    near_result = _probe(
        state, near, engine, cfg,
        episode=episode, condition=condition,
        b_delta=cfg.response.near_b_delta, support=STATE_LOWER,
    )
    far_result = _probe(
        state, far, engine, cfg,
        episode=episode, condition=condition,
        b_delta=cfg.response.far_b_delta, support=STATE_LOWER,
    )

    retained_state = decay_only(
        state, checkpoint_cfg.retention_interval_episodes, cfg.updates, counter
    )
    retention = _probe(
        retained_state, trained, engine, cfg,
        episode=episode, condition=condition, b_delta=STATE_LOWER, support=STATE_LOWER,
    )

    # Calibration is scored on the unaided probes only: a supported forecast is
    # not a forecast about the learner's own competence.
    confidence = np.column_stack(
        [unaided.confidence, near_result.confidence, far_result.confidence]
    )
    correct = np.column_stack([unaided.correct, near_result.correct, far_result.correct])
    brier = ((confidence - correct.astype(np.float64)) ** 2).mean(axis=1)
    ece = expected_calibration_error(confidence, correct, checkpoint_cfg.ece_bins)

    theta = state.theta(cfg.population)
    dependence = np.mean(
        [
            support_request_probability(
                theta, np.full(state.n_learners, unit.b_u, dtype=np.float64), state.D, cfg.response
            )
            for unit in trained
        ],
        axis=0,
    )

    frame = pd.DataFrame(
        {
            "learner_id": state.learner_id,
            "condition": condition,
            "checkpoint_episode": episode,
            "stratum_index": state.stratum_index,
            "unaided_accuracy_trained": unaided.accuracy,
            "supported_accuracy_trained": supported.accuracy,
            "support_gap": support_gap(supported.accuracy, unaided.accuracy),
            "near_transfer_accuracy": near_result.accuracy,
            "far_transfer_accuracy": far_result.accuracy,
            "retention_accuracy": retention.accuracy,
            "retention_interval_episodes": checkpoint_cfg.retention_interval_episodes,
            "brier_score": brier,
            "expected_calibration_error": ece,
            "dependence": dependence,
            "n_trained_items": len(trained),
            "n_near_items": len(near),
            "n_far_items": len(far),
            **{name: getattr(state, name) for name in STATE_COMPONENTS},
        }
    )
    return frame.loc[:, list(CHECKPOINT_COLUMNS)]


def support_gap(supported_accuracy: np.ndarray, unaided_accuracy: np.ndarray) -> np.ndarray:
    """Equation (26): supported accuracy minus unaided accuracy.

    Equation: `SupportGap = Accuracy^supported - Accuracy^unaided`.

    Direction: **larger means performance depends more on the support being
    there**. A large gap with a low unaided accuracy is the signature the study
    is looking for -- competence that exists only while the system is present.
    It is not a measure of learning, and a positive gap is expected in every
    condition because eq. (18) adds `omega * h` to the logit.
    """
    return np.asarray(supported_accuracy, dtype=np.float64) - np.asarray(
        unaided_accuracy, dtype=np.float64
    )


@dataclass(frozen=True)
class SeparationVerdict:
    """Whether two conditions are distinguishable once support is withdrawn.

    §10.6: if scaffolding and substitution are *not* distinguishable at
    checkpoints after support removal, that is a falsification signal about the
    model, not a result to pass over quietly. It is therefore recorded in the
    validation report either way.
    """

    metric: str
    checkpoint_episode: int
    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    difference: float
    threshold: float
    separated: bool

    def as_line(self) -> str:
        verdict = "SEPARATED" if self.separated else "NOT SEPARATED (§10.6 falsification signal)"
        return (
            f"| {self.checkpoint_episode} | {self.metric} | {self.condition_a} "
            f"{self.mean_a:.4f} | {self.condition_b} {self.mean_b:.4f} | "
            f"{self.difference:+.4f} | {self.threshold:.4f} | {verdict} |"
        )


def condition_separation(
    checkpoints: pd.DataFrame,
    cfg: LearnerConfig,
    *,
    metric: str = "unaided_accuracy_trained",
    condition_a: str = AI_SCAFFOLDING,
    condition_b: str = AI_SUBSTITUTION,
) -> Tuple[SeparationVerdict, ...]:
    """One verdict per checkpoint episode present in `checkpoints`.

    Returns an empty tuple when either condition is absent from the table --
    a run of one condition cannot separate two, and reporting `separated=False`
    for it would fabricate a falsification signal out of a missing arm.
    """
    if checkpoints.empty:
        return ()
    present = set(checkpoints["condition"].unique())
    if not {condition_a, condition_b} <= present:
        return ()
    verdicts = []
    threshold = cfg.checkpoints.separation_threshold
    for episode in sorted(checkpoints["checkpoint_episode"].unique()):
        window = checkpoints.loc[checkpoints["checkpoint_episode"] == episode]
        mean_a = float(window.loc[window["condition"] == condition_a, metric].mean())
        mean_b = float(window.loc[window["condition"] == condition_b, metric].mean())
        difference = mean_a - mean_b
        verdicts.append(
            SeparationVerdict(
                metric=metric,
                checkpoint_episode=int(episode),
                condition_a=condition_a,
                condition_b=condition_b,
                mean_a=mean_a,
                mean_b=mean_b,
                difference=difference,
                threshold=threshold,
                separated=bool(abs(difference) >= threshold),
            )
        )
    return tuple(verdicts)


def stratum_monotonicity(checkpoints: pd.DataFrame, metric: str) -> Dict[int, float]:
    """Mean `metric` per prior-knowledge stratum, for the §10.2 monotonicity check."""
    if metric not in checkpoints.columns:
        raise ValueError(f"checkpoint table has no column {metric!r}")
    grouped = checkpoints.groupby("stratum_index")[metric].mean()
    return {int(index): float(value) for index, value in grouped.items()}


def _bounds_check(frame: pd.DataFrame) -> None:  # pragma: no cover - diagnostic helper
    for column in ("unaided_accuracy_trained", "brier_score", "expected_calibration_error"):
        values = frame[column].to_numpy(dtype=np.float64)
        if values.min() < STATE_LOWER or values.max() > STATE_UPPER:
            raise ValueError(f"{column} escaped [{STATE_LOWER}, {STATE_UPPER}]")
