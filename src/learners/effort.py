"""Learner effort and instructional effectiveness (§7.4, eq. 19-20).

    E_iut = logistic(a0 + a1*Attempt + a2*Retrieval + a3*Explanation
                     - a4*AnswerProvided)
    F_iut = logistic(f0 + f1*Correct_u + f2*Coverage_u + f3*Adaptation
                     - f4*Mismatch)

`E_iut` is the quantity Phase IV multiplies by `Z(u,c,p)` in the plasticity
update `N <- (1-delta)N + eta * E * Z`. **Phase III ends here.** Nothing in this
package computes a neural quantity or reads a TRIBE artefact -- the two tracks
are parallel, and their product is the project's hypothesis rather than a
pipeline stage.

Operational proxies
-------------------
The brief specifies the equations but not their inputs: §7.4 says only that
"operational proxies should be derived from the tutor interaction". Every input
below is therefore a **definition made here**, and every one is an observable of
the interaction -- never a free parameter, so none of them can be tuned to
produce a result.

============================  ==============  =======================================================
input                         range           definition
============================  ==============  =======================================================
`Attempt`                     {0, 1}          the learner produced an independent attempt
`Retrieval`                   {0, 0.5, 1}     `Attempt * (1 + unaided_correct) / 2` -- an unaided
                                              attempt is a retrieval event, a successful one is a
                                              stronger retrieval event
`Explanation`                 [0, 1]          `Attempt * (1 - hint_depth / max_hint_depth)` -- the
                                              share of the solution the learner generated
`AnswerProvided`              {0, 1}          the solution was handed over
`Offloading`                  [0, 1]          `1 - Explanation` -- the share of the solution the
                                              system generated (eq. 23)
`Correct_u`                   [0, 1]          §5.4 correctness of the unit's content; from the units
                                              table, else the configured default **with a warning**
`Coverage_u`                  [0, 1]          §5.4 semantic coverage; same rule
`Adaptation`                  [0, 1]          `SupportOutcome.adaptation` -- how well the hint
                                              targeted the learner's actual error
`Mismatch`                    [0, 1]          `SupportOutcome.mismatch` -- difficulty misfit
`CorrectAfterError`           {0, 1}          wrong unaided, then right under support **without the
                                              answer being handed over** (eq. 22)
`TransferSuccess`             {0, 1}          the unaided transfer question was answered correctly
                                              (eq. 23)
`SupportUsed`                 [0, 1]          the realised support level `h` (eq. 25)
`SupportFaded`                [0, 1]          share of the offered support that was withdrawn (eq. 25)
`IndependentSuccess`          {0, 1}          `Attempt * unaided_correct` (eq. 25)
============================  ==============  =======================================================

`CorrectAfterError` excludes episodes where the answer was revealed on purpose:
being shown a solution and then reproducing it is not the same event as working
past one's own error, and counting it as such would let the substitution
condition accumulate memory strength for reading.

Directions, for the gate-19 discipline applied to behavioural quantities:
`a1..a3 > 0` and `a4 > 0` entering negatively, so **attempting, retrieving and
explaining raise effort, while being handed the answer lowers it**. `f1..f3 > 0`
and `f4 > 0` entering negatively, so **correct, well-covered, well-targeted
instruction raises effectiveness and difficulty misfit lowers it**.

Track B. numpy and pandas; never torch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from src.learners.config import (
    STATE_LOWER,
    STATE_UPPER,
    EffectivenessConfig,
    EffortConfig,
)
from src.learners.responses import logistic

#: Units-table columns feeding eq. (20), and the config field holding each one's
#: fallback. A missing column is defaulted *and reported* -- never silently.
EFFECTIVENESS_COLUMNS: Tuple[Tuple[str, str], ...] = (
    ("coverage", "coverage_default"),
    ("correctness", "correctness_default"),
)


def resolve_effectiveness_columns(
    units: pd.DataFrame, cfg: EffectivenessConfig
) -> Tuple[pd.DataFrame, List[str]]:
    """Attach `coverage` and `correctness`, reporting every substitution.

    `Coverage_u` comes from the Phase I corpus validation output (§5.4) when the
    units table carries it. When it does not, the configured default is used
    **and a warning naming the missing column is raised and returned**, so the
    substitution appears in the run log rather than only in the numbers.

    Returns
    -------
    (pandas.DataFrame, list of str)
        The frame with both columns present, and one message per defaulted
        column for `outputs/logs/phase3_run_<timestamp>.json`.
    """
    out = units.copy()
    messages: List[str] = []
    for column, default_field in EFFECTIVENESS_COLUMNS:
        default = float(getattr(cfg, default_field))
        if column not in out.columns:
            message = (
                f"units table has no {column!r} column: eq. (20) input defaulted to {default} "
                f"from effectiveness.{default_field}. This column is produced by the Phase I "
                f"§5.4 corpus validation; until that output is available the eq. (20) term is a "
                f"constant and cannot contribute any between-unit variation."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            messages.append(message)
            out[column] = default
            continue
        values = pd.to_numeric(out[column], errors="coerce")
        if values.isna().any():
            n_missing = int(values.isna().sum())
            message = (
                f"units table column {column!r} has {n_missing} missing or non-numeric value(s); "
                f"those rows defaulted to {default} from effectiveness.{default_field}."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            messages.append(message)
            values = values.fillna(default)
        if float(values.min()) < STATE_LOWER or float(values.max()) > STATE_UPPER:
            raise ValueError(
                f"units table column {column!r} must lie in [{STATE_LOWER}, {STATE_UPPER}]; got "
                f"[{values.min()}, {values.max()}]. It enters eq. (20) as a proportion."
            )
        out[column] = values.astype(np.float64)
    return out, messages


@dataclass(frozen=True)
class InteractionProxies:
    """Every observable the state updates and eq. (19)-(20) read.

    One place where all of the brief's unspecified inputs are defined, so that
    two equations cannot end up disagreeing about what `Retrieval` means.
    Every field is an array over learners.
    """

    attempt: np.ndarray
    retrieval: np.ndarray
    explanation: np.ndarray
    answer_provided: np.ndarray
    offloading: np.ndarray
    adaptation: np.ndarray
    mismatch: np.ndarray
    coverage: np.ndarray
    correctness: np.ndarray
    correct_after_error: np.ndarray
    transfer_success: np.ndarray
    support_used: np.ndarray
    support_faded: np.ndarray
    independent_success: np.ndarray

    _UNIT_INTERVAL = (
        "attempt",
        "retrieval",
        "explanation",
        "answer_provided",
        "offloading",
        "adaptation",
        "mismatch",
        "coverage",
        "correctness",
        "correct_after_error",
        "transfer_success",
        "support_used",
        "support_faded",
        "independent_success",
    )

    def __post_init__(self) -> None:
        shapes = {name: np.asarray(getattr(self, name)).shape for name in self._UNIT_INTERVAL}
        if len(set(shapes.values())) != 1:
            raise ValueError(f"every InteractionProxies array must have the same shape; got {shapes}")
        for name in self._UNIT_INTERVAL:
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"InteractionProxies.{name} contains non-finite values")
            if values.min() < STATE_LOWER or values.max() > STATE_UPPER:
                raise ValueError(
                    f"InteractionProxies.{name} must lie in [{STATE_LOWER}, {STATE_UPPER}] -- "
                    f"every proxy is a proportion or an indicator, by construction; got "
                    f"[{values.min()}, {values.max()}]"
                )

    @property
    def n_learners(self) -> int:
        return int(np.asarray(self.attempt).size)


def retrieval_proxy(attempt: np.ndarray, unaided_correct: np.ndarray) -> np.ndarray:
    """The eq. (19)/(22) `Retrieval` input, in {0, 0.5, 1}.

    Equation: `Attempt * (1 + unaided_correct) / 2`.

    Direction: increasing in both attempting and succeeding, so **more unaided
    retrieval raises effort (eq. 19) and consolidates memory (eq. 22)**.

    An unaided attempt is a retrieval event whether or not it succeeds -- which is
    why a failed attempt scores 0.5 rather than 0. A learner who never attempted
    performed no retrieval at all.

    Not collinear with `Attempt`: it varies among learners who did attempt, which
    is what keeps `a1` and `a2` in eq. (19) separately identifiable.

    ASSUMPTION: `Retrieval` appears in eq. (19) and eq. (22) and is defined in
    neither. One definition is used in both.
    """
    made = np.asarray(attempt, dtype=np.float64)
    correct = np.asarray(unaided_correct, dtype=np.float64)
    return made * (1 + correct) / 2


def explanation_proxy(
    attempt: np.ndarray, hint_depth: np.ndarray, max_hint_depth: int
) -> np.ndarray:
    """Share of the solution the learner generated, in [0, 1].

    Equation: `Attempt * (1 - hint_depth / max_hint_depth)`.

    Direction: **more hints consumed means a smaller learner-generated share**.
    A learner who was handed the whole worked solution scores 0, and so does one
    who never attempted.

    ASSUMPTION: §7.4 names "proportion of solution tokens generated by the
    learner" as the intended proxy. No tokens exist in a structured-descriptor
    engine, so hint depth stands in for them. Recorded in
    `docs/phase3_assumptions.md`.
    """
    if max_hint_depth < 1:
        raise ValueError(f"max_hint_depth must be >= 1; got {max_hint_depth}")
    made = np.asarray(attempt, dtype=np.float64)
    depth = np.asarray(hint_depth, dtype=np.float64)
    share = STATE_UPPER - depth / max_hint_depth
    return made * np.clip(share, STATE_LOWER, STATE_UPPER)


def offloading_proxy(explanation: np.ndarray) -> np.ndarray:
    """The eq. (23) `Offloading` input, in [0, 1].

    Equation: `Offloading = 1 - Explanation`.

    Direction: 1 means the system produced the whole solution, 0 means the
    learner did. It enters eq. (23) negatively, so **offloading erodes
    independent reasoning**.

    Defined as the exact complement of `Explanation` rather than as a separate
    quantity, so the two cannot drift apart and no second parameter is
    introduced.

    ASSUMPTION: `Offloading` appears only inside eq. (23); the brief never
    defines it.
    """
    return STATE_UPPER - np.asarray(explanation, dtype=np.float64)


def correct_after_error_proxy(
    unaided_correct: np.ndarray,
    supported_correct: np.ndarray,
    answer_provided: np.ndarray,
    attempt: np.ndarray,
) -> np.ndarray:
    """The eq. (22) `CorrectAfterError` input, in {0, 1}.

    Definition: the learner attempted, was wrong unaided, then answered
    correctly under support, **and the answer was not handed over**.

    Direction: it adds to memory strength in eq. (22), so **working past one's
    own error consolidates**. Reproducing a revealed solution does not count, and
    that exclusion is the whole reason the term is defined this narrowly.

    ASSUMPTION: `CorrectAfterError` appears only inside eq. (22).
    """
    attempted = np.asarray(attempt, dtype=bool)
    was_wrong = attempted & ~np.asarray(unaided_correct, dtype=bool)
    recovered = was_wrong & np.asarray(supported_correct, dtype=bool)
    return (recovered & ~np.asarray(answer_provided, dtype=bool)).astype(np.float64)


def independent_success_proxy(attempt: np.ndarray, unaided_correct: np.ndarray) -> np.ndarray:
    """The eq. (25) `IndependentSuccess` input, in {0, 1}.

    Equation: `IndependentSuccess = Attempt AND unaided_correct`.

    Direction: multiplied by `SupportFaded` in eq. (25), it is the only term that
    reduces dependence, so **succeeding without support that was withdrawn is
    what makes a learner less dependent**.

    ASSUMPTION: `IndependentSuccess` appears only inside eq. (25).
    """
    return (
        np.asarray(attempt, dtype=bool) & np.asarray(unaided_correct, dtype=bool)
    ).astype(np.float64)


def instructional_effort(proxies: InteractionProxies, cfg: EffortConfig) -> np.ndarray:
    """Equation (19): learner effort `E_iut`, in (0, 1).

    Equation: `logistic(a0 + a1*Attempt + a2*Retrieval + a3*Explanation
    - a4*AnswerProvided)`.

    Direction: increasing in `Attempt`, `Retrieval` and `Explanation`; decreasing
    in `AnswerProvided`. `E` is the term Phase IV multiplies by the predicted
    cortical response, so a stimulus consumed with no effort leaves no
    longitudinal trace no matter how strongly it is predicted to engage cortex.
    """
    logit = (
        cfg.a0
        + cfg.a1 * np.asarray(proxies.attempt, dtype=np.float64)
        + cfg.a2 * np.asarray(proxies.retrieval, dtype=np.float64)
        + cfg.a3 * np.asarray(proxies.explanation, dtype=np.float64)
        - cfg.a4 * np.asarray(proxies.answer_provided, dtype=np.float64)
    )
    if not np.isfinite(logit).all():
        raise ValueError("eq. (19) effort logit is non-finite; check the effort coefficients")
    return logistic(logit)


def instructional_effectiveness(
    proxies: InteractionProxies, cfg: EffectivenessConfig
) -> np.ndarray:
    """Equation (20): instructional effectiveness `F_iut`, in (0, 1).

    Equation: `logistic(f0 + f1*Correct_u + f2*Coverage_u + f3*Adaptation
    - f4*Mismatch)`.

    Direction: increasing in unit correctness, semantic coverage and how well the
    support targeted the learner's error; decreasing in difficulty misfit.
    """
    logit = (
        cfg.f0
        + cfg.f1 * np.asarray(proxies.correctness, dtype=np.float64)
        + cfg.f2 * np.asarray(proxies.coverage, dtype=np.float64)
        + cfg.f3 * np.asarray(proxies.adaptation, dtype=np.float64)
        - cfg.f4 * np.asarray(proxies.mismatch, dtype=np.float64)
    )
    if not np.isfinite(logit).all():
        raise ValueError(
            "eq. (20) effectiveness logit is non-finite; check the effectiveness coefficients"
        )
    return logistic(logit)
