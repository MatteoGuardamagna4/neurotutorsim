"""One instructional episode (§3.1), for every learner at once.

    problem -> first response -> instructional intervention -> second response
            -> transfer question

The order of operations is fixed by §3.1 and **must not be rearranged**:

1. First unaided response (`h = 0`).
2. The tutor policy observes correctness and produces a `SupportOutcome`.
3. Second response under support level `h`.
4. Transfer question, unaided.
5. Effort and effectiveness computed from the realised interaction.
6. State update.

Steps 1 and 2 in that order are the whole point: a tutor that acted before
seeing an attempt would be the `ai_substitution` condition by accident, and the
contrast the study rests on would collapse. `EpisodeBatch` refuses a `first` or
`transfer` stage carrying support, so the ordering is enforced rather than
merely documented.

`episode_id` is constant across conditions for the same unit, so a within-unit
contrast joins on it.

The hint ladder without a per-learner loop
------------------------------------------
"Answer revealed after the ladder is exhausted" is a per-learner amount of
interaction. It is resolved by looping over **hint depths** -- at most three --
and masking, never over learners. Every learner is scored at every depth and the
mask decides whose result is kept, which also means the RNG stream advances by
the same amount regardless of who answered what. That is what keeps two runs
with the same master seed bitwise identical.

Track B. numpy and pandas; never torch.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from src.learners import effort as effort_module
from src.learners import support as support_module
from src.learners.config import LearnerConfig
from src.learners.curriculum import UnitView
from src.learners.effort import InteractionProxies
from src.learners.engine import (
    STAGE_FIRST,
    STAGE_SUPPORTED,
    STAGE_TRANSFER,
    EpisodeBatch,
    ResponseEngine,
)
from src.learners.population import LearnerState
from src.learners.responses import latency_proxy
from src.learners.updates import ClipCounter, accumulate_brier, apply_updates

#: `data/processed/responses.csv` columns.
#:
#: `stage` and `episode_id` are **additions** to the §4.2 schema. §3.1 scores
#: three responses per learner per episode (first unaided, supported, transfer),
#: so `learner_id + episode + condition` does not identify a row: without
#: `stage` the table has no key, and collapsing the three events into one would
#: destroy the supported-minus-unaided information eq. (26) is built from.
#: Recorded as an addition in `docs/data_dictionary_phase3.md`.
RESPONSE_COLUMNS: Tuple[str, ...] = (
    "learner_id",
    "episode",
    "episode_id",
    "condition",
    "stage",
    "prompt",
    "response",
    "answer",
    "confidence",
    "correctness",
    "latency_proxy",
    "token_count",
)

#: `data/processed/learner_state.parquet` columns (§4.2, unchanged).
STATE_COLUMNS: Tuple[str, ...] = (
    "learner_id",
    "time",
    "K",
    "M",
    "R",
    "C",
    "D",
    "effort",
    "support",
    "correctness",
)

#: Response descriptors used by the logistic engine in place of free text.
RESPONSE_CORRECT = "correct"
RESPONSE_MISCONCEPTION = "misconception"
RESPONSE_INCORRECT = "incorrect"
ANSWER_UNKNOWN = "unspecified"

_FIELD_SEPARATOR = "|"


def episode_id(episode: int, unit: UnitView) -> str:
    """Identifier shared by the same unit at the same episode index in every condition."""
    return f"ep{episode:04d}{_FIELD_SEPARATOR}{unit.unit_id}"


@dataclass(frozen=True)
class EpisodeRecords:
    """Column-oriented output of one episode, ready to append to a chunk.

    Column arrays rather than row dicts: at 5,000 learners x 3 response stages an
    episode emits 15,000 rows, and building them as Python objects would cost
    more than the entire simulation.
    """

    responses: Dict[str, np.ndarray]
    state: Dict[str, np.ndarray]
    #: Per-episode diagnostics for the run log and the validation report.
    diagnostics: Dict[str, float]

    @property
    def n_response_rows(self) -> int:
        return int(self.responses["learner_id"].size)


def _descriptor(unit: UnitView, condition: str, stage: str, support: np.ndarray) -> np.ndarray:
    """Structured prompt descriptor, one per learner.

    The logistic engine has no prompt. §4.2 requires the column, so it carries a
    compact, parseable descriptor of *what was asked* --
    `unit|condition|stage|d=<difficulty>|h=<support>` -- rather than free text
    invented to fill a string field. Documented in
    `docs/data_dictionary_phase3.md`.
    """
    prefix = (
        f"{unit.unit_id}{_FIELD_SEPARATOR}{condition}{_FIELD_SEPARATOR}{stage}"
        f"{_FIELD_SEPARATOR}d={unit.difficulty}{_FIELD_SEPARATOR}h="
    )
    return np.char.add(prefix, np.char.mod("%.2f", np.asarray(support, dtype=np.float64)))


def _response_and_answer(
    unit: UnitView, correct: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Response and answer descriptors for one scored stage.

    A wrong answer is labelled `misconception` when the unit documents one --
    that is the error mode the corpus declares -- and `incorrect` otherwise. No
    answer is invented: a unit with no `reference_answer` column yields
    `unspecified`.
    """
    wrong_label = RESPONSE_MISCONCEPTION if unit.misconception_answer else RESPONSE_INCORRECT
    right_answer = unit.reference_answer if unit.reference_answer is not None else ANSWER_UNKNOWN
    wrong_answer = (
        unit.misconception_answer if unit.misconception_answer is not None else ANSWER_UNKNOWN
    )
    hit = np.asarray(correct, dtype=bool)
    return (
        np.where(hit, RESPONSE_CORRECT, wrong_label),
        np.where(hit, str(right_answer), str(wrong_answer)),
    )


def _token_count(prompt: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Descriptor fields, not LLM tokens.

    The column exists so the schema is identical whichever engine produced the
    row. For the logistic engine it counts the `|`-separated fields of the prompt
    descriptor plus the response descriptor; for a scoring LLM engine it would be
    the real prompt length. `docs/data_dictionary_phase3.md` says so explicitly,
    because a column named `token_count` that silently means something else in
    half the rows is worse than no column.
    """
    prompt_fields = np.char.count(prompt, _FIELD_SEPARATOR) + 1
    response_fields = np.char.count(response, _FIELD_SEPARATOR) + 1
    return (prompt_fields + response_fields).astype(np.int64)


def run_episode(
    state: LearnerState,
    unit: UnitView,
    condition: str,
    engine: ResponseEngine,
    cfg: LearnerConfig,
    rng: np.random.Generator,
    *,
    episode: int,
    counter: ClipCounter,
) -> Tuple[LearnerState, EpisodeRecords]:
    """Run one episode for every learner and advance the state.

    Parameters
    ----------
    state
        Current learner state. **Advanced in place**; the returned state *is*
        this object. At §9.3 volumes copying the state 1.2e10 times is not an
        option, so the mutation is explicit rather than hidden behind a copy.
        Checkpoint assessments never call this -- they work on `state.copy()`.
    unit
        The unit taught this episode.
    condition
        One of the §3.2 instructional conditions.
    engine
        Anything satisfying `ResponseEngine`.
    cfg
        Full learner configuration.
    rng
        The `tutor` stream generator, used for the tutor's error diagnosis. The
        engine holds its own `responses` stream.
    episode
        0-based episode index, written into the records as `time`.
    counter
        Clip counter, accumulated across the run.

    Returns
    -------
    (LearnerState, EpisodeRecords)
        The advanced state and this episode's records.
    """
    support_cfg = cfg.support
    n_learners = state.n_learners
    theta = state.theta(cfg.population)
    b_u = np.full(n_learners, unit.b_u, dtype=np.float64)
    no_support = np.zeros(n_learners, dtype=np.float64)
    identifier = episode_id(episode, unit)

    def _batch(stage: str, difficulty: np.ndarray, support: np.ndarray) -> EpisodeBatch:
        return EpisodeBatch(
            unit_id=unit.unit_id,
            episode=episode,
            condition=condition,
            stage=stage,
            b_u=difficulty,
            theta=theta,
            memory=state.M,
            reasoning=state.R,
            dependence=state.D,
            support_level=support,
            confidence_bias=state.confidence_bias,
        )

    # -- 1. first unaided response ------------------------------------------
    first = engine.respond(_batch(STAGE_FIRST, b_u, no_support))
    if support_module.requires_attempt(condition):
        attempt = np.ones(n_learners, dtype=bool)
    else:
        # Only `ai_substitution` reaches here: no attempt is required, so a
        # learner who asks for help receives it before producing any work.
        attempt = ~np.asarray(first.support_requested, dtype=bool)
    unaided_correct = np.asarray(first.answer_correct, dtype=bool) & attempt

    # -- 2./3. tutor intervention and the supported response ----------------
    policy = support_module.get_policy(condition)
    history = support_module.AttemptHistory(
        n_attempts=attempt.astype(np.int64),
        n_hints=np.zeros(n_learners, dtype=np.int64),
        unaided_correct=unaided_correct,
        resolved=unaided_correct.copy(),
        independent_success_streak=state.independent_success_streak,
    )

    resolved = unaided_correct.copy()
    h_realised = no_support.copy()
    hint_depth = np.zeros(n_learners, dtype=np.int64)
    answer_provided = np.zeros(n_learners, dtype=bool)
    supported_correct = np.zeros(n_learners, dtype=bool)
    supported_confidence = np.zeros(n_learners, dtype=np.float64)
    supported_scored = np.zeros(n_learners, dtype=bool)
    adaptation: np.ndarray | None = None
    mismatch: np.ndarray | None = None
    h_nominal: np.ndarray | None = None

    for _ in range(support_module.max_steps(condition, support_cfg)):
        plan = policy(theta, history, unit, support_cfg, rng=rng)
        if adaptation is None:
            # Recorded for every learner from the first step, including those who
            # never needed a hint: `adaptation` and `mismatch` describe the
            # instruction the condition *offers*, while `h` and `hint_depth`
            # describe what was consumed.
            adaptation = np.asarray(plan.adaptation, dtype=np.float64).copy()
            mismatch = np.asarray(plan.mismatch, dtype=np.float64).copy()
            h_nominal = np.asarray(plan.h_nominal, dtype=np.float64).copy()
        active = ~resolved
        step = engine.respond(_batch(STAGE_SUPPORTED, b_u, plan.h))
        step_correct = np.asarray(step.answer_correct, dtype=bool) | np.asarray(
            plan.answer_provided, dtype=bool
        )
        h_realised = np.where(active, plan.h, h_realised)
        hint_depth = np.where(active, plan.hint_depth, hint_depth)
        answer_provided = np.where(active, plan.answer_provided, answer_provided)
        adaptation = np.where(active, plan.adaptation, adaptation)
        supported_correct = np.where(active, step_correct, supported_correct)
        supported_confidence = np.where(active, step.confidence, supported_confidence)
        supported_scored = supported_scored | active
        resolved = resolved | (active & step_correct)
        history = replace(
            history,
            n_hints=np.where(active, history.n_hints + 1, history.n_hints).astype(np.int64),
            resolved=resolved,
        )

    assert adaptation is not None and mismatch is not None and h_nominal is not None

    # -- 4. transfer question, unaided --------------------------------------
    transfer = engine.respond(
        _batch(STAGE_TRANSFER, b_u + cfg.response.transfer_b_delta, no_support)
    )
    transfer_correct = np.asarray(transfer.answer_correct, dtype=bool)

    # -- 5. effort and effectiveness from the realised interaction -----------
    attempt_f = attempt.astype(np.float64)
    explanation = effort_module.explanation_proxy(
        attempt_f, hint_depth, support_cfg.max_hint_depth
    )
    proxies = InteractionProxies(
        attempt=attempt_f,
        retrieval=effort_module.retrieval_proxy(attempt_f, unaided_correct),
        explanation=explanation,
        answer_provided=answer_provided.astype(np.float64),
        offloading=effort_module.offloading_proxy(explanation),
        adaptation=adaptation,
        mismatch=mismatch,
        coverage=np.full(n_learners, unit.coverage, dtype=np.float64),
        correctness=np.full(n_learners, unit.correctness, dtype=np.float64),
        correct_after_error=effort_module.correct_after_error_proxy(
            unaided_correct, supported_correct, answer_provided, attempt
        ),
        transfer_success=transfer_correct.astype(np.float64),
        support_used=h_realised,
        support_faded=support_module.support_faded(h_realised, h_nominal),
        independent_success=effort_module.independent_success_proxy(attempt_f, unaided_correct),
    )
    learner_effort = effort_module.instructional_effort(proxies, cfg.effort)
    effectiveness = effort_module.instructional_effectiveness(proxies, cfg.effectiveness)

    # -- 6. state update ----------------------------------------------------
    # Calibration is scored on the responses that are genuine forecasts: the two
    # unaided ones. A supported response whose answer was revealed is not a
    # forecast, and counting it would reward the substitution condition for
    # confidence in a solution it was handed.
    forecast_confidence = np.column_stack([first.confidence, transfer.confidence])
    forecast_correct = np.column_stack([unaided_correct, transfer_correct])
    forecast_mask = np.column_stack([attempt, np.ones(n_learners, dtype=bool)])
    apply_updates(
        state,
        proxies,
        learner_effort,
        effectiveness,
        forecast_confidence,
        forecast_correct,
        cfg.updates,
        counter,
        mask=forecast_mask,
    )

    records = _build_records(
        state=state,
        unit=unit,
        condition=condition,
        episode=episode,
        identifier=identifier,
        attempt=attempt,
        unaided_correct=unaided_correct,
        first_confidence=np.asarray(first.confidence, dtype=np.float64),
        supported_scored=supported_scored,
        supported_correct=supported_correct,
        supported_confidence=supported_confidence,
        transfer_correct=transfer_correct,
        transfer_confidence=np.asarray(transfer.confidence, dtype=np.float64),
        hint_depth=hint_depth,
        h_realised=h_realised,
        learner_effort=learner_effort,
        effectiveness=effectiveness,
        proxies=proxies,
        cfg=cfg,
    )
    return state, records


def _build_records(
    *,
    state: LearnerState,
    unit: UnitView,
    condition: str,
    episode: int,
    identifier: str,
    attempt: np.ndarray,
    unaided_correct: np.ndarray,
    first_confidence: np.ndarray,
    supported_scored: np.ndarray,
    supported_correct: np.ndarray,
    supported_confidence: np.ndarray,
    transfer_correct: np.ndarray,
    transfer_confidence: np.ndarray,
    hint_depth: np.ndarray,
    h_realised: np.ndarray,
    learner_effort: np.ndarray,
    effectiveness: np.ndarray,
    proxies: InteractionProxies,
    cfg: LearnerConfig,
) -> EpisodeRecords:
    """Assemble the two output tables plus the episode's diagnostics.

    A response row is emitted only for a response that happened: a learner who
    never attempted has no unaided row, and one who was right first time has no
    supported row. Padding the table with placeholder rows would put responses
    that were never scored into every accuracy denominator.
    """
    n_learners = state.n_learners
    zeros = np.zeros(n_learners, dtype=np.float64)
    stages: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [
        (STAGE_FIRST, attempt, unaided_correct, first_confidence, zeros, np.zeros(n_learners, dtype=np.int64)),
        (
            STAGE_SUPPORTED,
            supported_scored,
            supported_correct,
            supported_confidence,
            h_realised,
            hint_depth,
        ),
        (
            STAGE_TRANSFER,
            np.ones(n_learners, dtype=bool),
            transfer_correct,
            transfer_confidence,
            zeros,
            np.zeros(n_learners, dtype=np.int64),
        ),
    ]

    columns: Dict[str, List[np.ndarray]] = {name: [] for name in RESPONSE_COLUMNS}
    for stage, mask, correct, confidence, support, depth in stages:
        keep = np.asarray(mask, dtype=bool)
        if not keep.any():
            continue
        prompt = _descriptor(unit, condition, stage, support[keep])
        response, answer = _response_and_answer(unit, correct[keep])
        columns["learner_id"].append(state.learner_id[keep])
        columns["episode"].append(np.full(int(keep.sum()), episode, dtype=np.int64))
        columns["episode_id"].append(np.full(int(keep.sum()), identifier, dtype=object))
        columns["condition"].append(np.full(int(keep.sum()), condition, dtype=object))
        columns["stage"].append(np.full(int(keep.sum()), stage, dtype=object))
        columns["prompt"].append(prompt)
        columns["response"].append(response)
        columns["answer"].append(answer)
        columns["confidence"].append(np.asarray(confidence, dtype=np.float64)[keep])
        columns["correctness"].append(np.asarray(correct, dtype=bool)[keep].astype(np.int64))
        columns["latency_proxy"].append(
            latency_proxy(
                state.speed_factor[keep],
                np.asarray(depth, dtype=np.float64)[keep],
                np.asarray(attempt, dtype=np.float64)[keep],
                cfg.response,
            )
        )
        columns["token_count"].append(_token_count(prompt, response))

    responses = {name: np.concatenate(parts) for name, parts in columns.items() if parts}

    state_table = {
        "learner_id": state.learner_id,
        "time": np.full(n_learners, episode, dtype=np.int64),
        "K": state.K,
        "M": state.M,
        "R": state.R,
        "C": state.C,
        "D": state.D,
        "effort": np.asarray(learner_effort, dtype=np.float64),
        "support": np.asarray(h_realised, dtype=np.float64),
        # The episode's *unaided* correctness: the one response in the episode
        # that is not confounded by the support level.
        "correctness": np.asarray(unaided_correct, dtype=bool).astype(np.int64),
    }

    diagnostics = {
        "episode": float(episode),
        "mean_effort": float(np.mean(learner_effort)),
        "mean_effectiveness": float(np.mean(effectiveness)),
        "mean_support": float(np.mean(h_realised)),
        "mean_unaided_correct": float(np.mean(unaided_correct)),
        "mean_transfer_correct": float(np.mean(transfer_correct)),
        "mean_answer_provided": float(np.mean(proxies.answer_provided)),
        "mean_offloading": float(np.mean(proxies.offloading)),
        "attempt_rate": float(np.mean(attempt)),
    }
    return EpisodeRecords(responses=responses, state=state_table, diagnostics=diagnostics)
