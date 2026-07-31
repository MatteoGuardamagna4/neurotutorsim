"""Equations (19)-(20) and the operational proxies the brief leaves undefined.

§7.4 says only that "operational proxies should be derived from the tutor
interaction". Every proxy is therefore a definition made in `effort.py`, and the
property these tests protect is that each one is an **observable** with a fixed
range -- never a free parameter that could be tuned until the conditions
separate.
"""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pytest

from src.learners.config import STATE_LOWER, STATE_UPPER
from src.learners.effort import (
    EFFECTIVENESS_COLUMNS,
    InteractionProxies,
    correct_after_error_proxy,
    explanation_proxy,
    independent_success_proxy,
    instructional_effectiveness,
    instructional_effort,
    offloading_proxy,
    resolve_effectiveness_columns,
    retrieval_proxy,
)

N = 6


def make_proxies(**overrides) -> InteractionProxies:
    base = {
        "attempt": np.ones(N),
        "retrieval": np.full(N, 0.5),
        "explanation": np.full(N, 0.5),
        "answer_provided": np.zeros(N),
        "offloading": np.full(N, 0.5),
        "adaptation": np.full(N, 0.5),
        "mismatch": np.full(N, 0.3),
        "coverage": np.ones(N),
        "correctness": np.ones(N),
        "correct_after_error": np.zeros(N),
        "transfer_success": np.full(N, 0.5),
        "support_used": np.full(N, 0.3),
        "support_faded": np.full(N, 0.5),
        "independent_success": np.full(N, 0.5),
    }
    base.update(overrides)
    return InteractionProxies(**base)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def test_retrieval_takes_three_documented_values():
    no_attempt = retrieval_proxy(np.zeros(N), np.zeros(N))
    failed = retrieval_proxy(np.ones(N), np.zeros(N))
    succeeded = retrieval_proxy(np.ones(N), np.ones(N))
    assert np.allclose(no_attempt, 0.0)
    assert np.allclose(failed, 0.5)
    assert np.allclose(succeeded, 1.0)


def test_a_failed_attempt_still_counts_as_retrieval():
    """Attempting is the effortful part; success is a stronger retrieval event."""
    failed = retrieval_proxy(np.ones(N), np.zeros(N))
    none = retrieval_proxy(np.zeros(N), np.zeros(N))
    assert np.all(failed > none)


def test_retrieval_is_not_collinear_with_attempt():
    """Otherwise a1 and a2 in eq. (19) would not be separately identifiable."""
    attempt = np.ones(20)
    correct = np.tile([0.0, 1.0], 10)
    retrieval = retrieval_proxy(attempt, correct)
    assert retrieval.std() > 0
    assert attempt.std() == 0


# ---------------------------------------------------------------------------
# Explanation and Offloading
# ---------------------------------------------------------------------------


def test_explanation_falls_as_hints_are_consumed():
    depths = np.arange(4, dtype=np.float64)
    values = explanation_proxy(np.ones(4), depths, 3)
    assert np.all(np.diff(values) <= 0)
    assert values[0] == pytest.approx(1.0)
    assert values[3] == pytest.approx(0.0)


def test_a_learner_who_never_attempted_generated_nothing():
    assert np.allclose(explanation_proxy(np.zeros(N), np.zeros(N), 3), 0.0)


def test_explanation_rejects_a_zero_depth_ladder():
    with pytest.raises(ValueError, match="max_hint_depth must be >= 1"):
        explanation_proxy(np.ones(N), np.zeros(N), 0)


def test_offloading_is_the_exact_complement_of_explanation():
    """Defined as the complement so the two cannot drift apart."""
    explanation = np.linspace(STATE_LOWER, STATE_UPPER, 11)
    np.testing.assert_allclose(offloading_proxy(explanation), STATE_UPPER - explanation)


def test_full_offloading_when_the_answer_is_handed_over():
    explanation = explanation_proxy(np.zeros(N), np.full(N, 3.0), 3)
    assert np.allclose(offloading_proxy(explanation), STATE_UPPER)


# ---------------------------------------------------------------------------
# CorrectAfterError -- the definition that excludes revealed answers
# ---------------------------------------------------------------------------


def test_correct_after_error_needs_an_error_then_a_recovery():
    attempt = np.ones(4)
    unaided = np.array([True, False, False, True])
    supported = np.array([True, True, False, False])
    answer = np.zeros(4, dtype=bool)
    values = correct_after_error_proxy(unaided, supported, answer, attempt)
    # only index 1 was wrong unaided and then right under support
    np.testing.assert_allclose(values, [0.0, 1.0, 0.0, 0.0])


def test_a_revealed_answer_does_not_count_as_recovery():
    """Reproducing a solution one was shown is not working past one's own error."""
    values = correct_after_error_proxy(
        np.zeros(N, dtype=bool),
        np.ones(N, dtype=bool),
        np.ones(N, dtype=bool),
        np.ones(N),
    )
    assert np.allclose(values, 0.0)


def test_a_learner_who_never_attempted_cannot_recover():
    values = correct_after_error_proxy(
        np.zeros(N, dtype=bool),
        np.ones(N, dtype=bool),
        np.zeros(N, dtype=bool),
        np.zeros(N),
    )
    assert np.allclose(values, 0.0)


def test_independent_success_needs_both_attempt_and_correctness():
    attempt = np.array([1.0, 1.0, 0.0, 0.0])
    correct = np.array([True, False, True, False])
    np.testing.assert_allclose(
        independent_success_proxy(attempt, correct), [1.0, 0.0, 0.0, 0.0]
    )


# ---------------------------------------------------------------------------
# InteractionProxies bounds
# ---------------------------------------------------------------------------


def test_out_of_range_proxy_raises():
    with pytest.raises(ValueError, match="every proxy is a proportion or an indicator"):
        make_proxies(adaptation=np.full(N, 1.5))


def test_non_finite_proxy_raises():
    values = np.full(N, 0.5)
    values[0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        make_proxies(mismatch=values)


def test_ragged_proxies_raise():
    with pytest.raises(ValueError, match="same shape"):
        make_proxies(coverage=np.ones(N + 1))


# ---------------------------------------------------------------------------
# Equation 19
# ---------------------------------------------------------------------------


def test_effort_rises_with_attempt_retrieval_and_explanation(learner_config):
    cfg = learner_config.effort
    low = instructional_effort(
        make_proxies(attempt=np.zeros(N), retrieval=np.zeros(N), explanation=np.zeros(N)), cfg
    )
    high = instructional_effort(
        make_proxies(attempt=np.ones(N), retrieval=np.ones(N), explanation=np.ones(N)), cfg
    )
    assert np.all(high > low)


def test_effort_falls_when_the_answer_is_provided(learner_config):
    cfg = learner_config.effort
    without = instructional_effort(make_proxies(answer_provided=np.zeros(N)), cfg)
    with_answer = instructional_effort(make_proxies(answer_provided=np.ones(N)), cfg)
    assert np.all(with_answer < without)


def test_effort_stays_inside_the_open_unit_interval(learner_config):
    values = instructional_effort(make_proxies(), learner_config.effort)
    assert values.min() > STATE_LOWER
    assert values.max() < STATE_UPPER


def test_a_substitution_shaped_interaction_collapses_effort(learner_config):
    """No attempt, no retrieval, nothing explained, answer handed over."""
    substitution = instructional_effort(
        make_proxies(
            attempt=np.zeros(N),
            retrieval=np.zeros(N),
            explanation=np.zeros(N),
            answer_provided=np.ones(N),
        ),
        learner_config.effort,
    )
    scaffolding = instructional_effort(
        make_proxies(
            attempt=np.ones(N),
            retrieval=np.ones(N),
            explanation=np.full(N, 0.6),
            answer_provided=np.zeros(N),
        ),
        learner_config.effort,
    )
    assert substitution.mean() < 0.2
    assert scaffolding.mean() > 0.6


# ---------------------------------------------------------------------------
# Equation 20
# ---------------------------------------------------------------------------


def test_effectiveness_rises_with_correctness_coverage_and_adaptation(learner_config):
    cfg = learner_config.effectiveness
    low = instructional_effectiveness(
        make_proxies(correctness=np.zeros(N), coverage=np.zeros(N), adaptation=np.zeros(N)), cfg
    )
    high = instructional_effectiveness(
        make_proxies(correctness=np.ones(N), coverage=np.ones(N), adaptation=np.ones(N)), cfg
    )
    assert np.all(high > low)


def test_effectiveness_falls_with_mismatch(learner_config):
    cfg = learner_config.effectiveness
    matched = instructional_effectiveness(make_proxies(mismatch=np.zeros(N)), cfg)
    mismatched = instructional_effectiveness(make_proxies(mismatch=np.ones(N)), cfg)
    assert np.all(mismatched < matched)


def test_adaptation_still_moves_effectiveness_at_the_defaults(learner_config):
    """F must not sit so near its ceiling that adaptation stops mattering.

    The traditional / scaffolding contrast exists *only* through adaptation, so a
    saturated F would collapse it for arithmetic reasons rather than substantive
    ones. This guards the f0..f2 retuning that fixed exactly that.
    """
    cfg = learner_config.effectiveness
    prewritten = instructional_effectiveness(
        make_proxies(adaptation=np.full(N, learner_config.support.adaptation_traditional)), cfg
    )
    diagnosed = instructional_effectiveness(
        make_proxies(
            adaptation=np.full(
                N,
                learner_config.support.adaptation_scaffolding_base
                + learner_config.support.adaptation_scaffolding_gain,
            )
        ),
        cfg,
    )
    assert (diagnosed - prewritten).mean() > 0.05


# ---------------------------------------------------------------------------
# Coverage / correctness resolution
# ---------------------------------------------------------------------------


def test_missing_coverage_column_warns_and_names_it(units_frame, learner_config):
    with pytest.warns(RuntimeWarning, match="no 'coverage' column"):
        resolved, messages = resolve_effectiveness_columns(
            units_frame, learner_config.effectiveness
        )
    assert (resolved["coverage"] == learner_config.effectiveness.coverage_default).all()
    assert any("coverage" in message for message in messages)
    assert any("§5.4 corpus validation" in message for message in messages)


def test_both_effectiveness_columns_are_reported(units_frame, learner_config):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, messages = resolve_effectiveness_columns(units_frame, learner_config.effectiveness)
    assert len(messages) == len(EFFECTIVENESS_COLUMNS)


def test_a_present_column_is_used_and_not_warned_about(units_frame, learner_config):
    units_frame = units_frame.copy()
    units_frame["coverage"] = 0.75
    units_frame["correctness"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        resolved, messages = resolve_effectiveness_columns(
            units_frame, learner_config.effectiveness
        )
    assert messages == []
    assert (resolved["coverage"] == 0.75).all()


def test_partially_missing_values_are_reported(units_frame, learner_config):
    units_frame = units_frame.copy()
    units_frame["coverage"] = 0.75
    units_frame.loc[0, "coverage"] = np.nan
    units_frame["correctness"] = 1.0
    with pytest.warns(RuntimeWarning, match="1 missing or non-numeric"):
        resolved, messages = resolve_effectiveness_columns(
            units_frame, learner_config.effectiveness
        )
    assert resolved.loc[0, "coverage"] == learner_config.effectiveness.coverage_default
    assert len(messages) == 1


def test_out_of_range_coverage_raises(units_frame, learner_config):
    units_frame = units_frame.copy()
    units_frame["coverage"] = 1.4
    units_frame["correctness"] = 1.0
    with pytest.raises(ValueError, match="must lie in"):
        resolve_effectiveness_columns(units_frame, learner_config.effectiveness)


# ---------------------------------------------------------------------------
# The §10.3 zero-effort control
# ---------------------------------------------------------------------------


def test_zeroed_effort_config_makes_effort_constant(learner_config):
    """With a1..a4 all zero, E cannot vary with the interaction at all."""
    zeroed = learner_config.effort.zeroed()
    substitution = instructional_effort(
        make_proxies(
            attempt=np.zeros(N),
            retrieval=np.zeros(N),
            explanation=np.zeros(N),
            answer_provided=np.ones(N),
        ),
        zeroed,
    )
    scaffolding = instructional_effort(
        make_proxies(
            attempt=np.ones(N),
            retrieval=np.ones(N),
            explanation=np.ones(N),
            answer_provided=np.zeros(N),
        ),
        zeroed,
    )
    np.testing.assert_allclose(substitution, scaffolding)


def test_keeping_a4_would_break_the_control(learner_config):
    """The finding behind zeroing a4 as well as a1..a3.

    The task spec writes the §10.3 control as `a1 = a2 = a3 = 0`. This test shows
    why that is not sufficient: `AnswerProvided` still separates substitution from
    the other conditions, so effort-mediated differences do not vanish.
    """
    partial = dataclasses.replace(learner_config.effort, a1=0.0, a2=0.0, a3=0.0)
    substitution = instructional_effort(make_proxies(answer_provided=np.ones(N)), partial)
    scaffolding = instructional_effort(make_proxies(answer_provided=np.zeros(N)), partial)
    assert not np.allclose(substitution, scaffolding)
