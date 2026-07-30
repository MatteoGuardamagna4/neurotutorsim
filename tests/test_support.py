"""Deterministic tutor policies (§3.4) and the persistence modifier.

The condition table in `support.py` is the specification; these tests are that
table in executable form. The one property to protect above the others: the
`traditional` and `ai_scaffolding` ladders must have the same depth and the same
attempt requirement, so that a condition contrast is a contrast in *adaptation*
and not in how much of the work the tutor did.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.learners.config import (
    AI_SCAFFOLDING,
    AI_SUBSTITUTION,
    STATE_LOWER,
    STATE_UPPER,
    STATIC_AI,
    TRADITIONAL,
)
from src.learners.support import (
    GRADUAL_FADING,
    IMMEDIATE_WITHDRAWAL,
    PERSISTENT,
    POLICIES,
    AttemptHistory,
    apply_persistence,
    difficulty_mismatch,
    get_policy,
    max_steps,
    requires_attempt,
    support_faded,
)

N = 8
ALL_CONDITIONS = (TRADITIONAL, AI_SCAFFOLDING, AI_SUBSTITUTION, STATIC_AI)


def make_history(*, n_hints=0, unaided_correct=False, streak=0):
    return AttemptHistory(
        n_attempts=np.ones(N, dtype=np.int64),
        n_hints=np.full(N, n_hints, dtype=np.int64),
        unaided_correct=np.full(N, unaided_correct, dtype=bool),
        resolved=np.full(N, unaided_correct, dtype=bool),
        independent_success_streak=np.full(N, streak, dtype=np.int64),
    )


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# The §3.4 condition table
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ALL_CONDITIONS)
def test_every_condition_has_a_policy(condition):
    assert condition in POLICIES
    assert callable(get_policy(condition))


def test_unknown_condition_raises_rather_than_defaulting():
    with pytest.raises(KeyError, match="no support policy registered"):
        get_policy("ai_socratic")


@pytest.mark.parametrize("condition", ALL_CONDITIONS)
def test_outcomes_are_in_bounds(condition, learner_config, unit_views, rng):
    outcome = get_policy(condition)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    for name in ("h", "adaptation", "mismatch", "h_nominal"):
        values = getattr(outcome, name)
        assert values.min() >= STATE_LOWER
        assert values.max() <= STATE_UPPER
    assert outcome.hint_depth.dtype == np.int64


def test_only_substitution_waives_the_attempt():
    assert requires_attempt(TRADITIONAL)
    assert requires_attempt(AI_SCAFFOLDING)
    assert requires_attempt(STATIC_AI)
    assert not requires_attempt(AI_SUBSTITUTION)


def test_substitution_gives_maximal_support_immediately(learner_config, unit_views, rng):
    outcome = get_policy(AI_SUBSTITUTION)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    assert outcome.answer_provided.all()
    assert not outcome.attempt_made.any()
    assert np.allclose(outcome.h_nominal, learner_config.support.substitution_h)
    assert (outcome.hint_depth == learner_config.support.max_hint_depth).all()


def test_traditional_withholds_the_answer_until_the_ladder_is_exhausted(
    learner_config, unit_views, rng
):
    depth = learner_config.support.max_hint_depth
    for n_hints in range(depth):
        outcome = get_policy(TRADITIONAL)(
            np.zeros(N), make_history(n_hints=n_hints), unit_views[0], learner_config.support, rng=rng
        )
        assert not outcome.answer_provided.any(), f"answer revealed at hint {n_hints}"
        assert outcome.attempt_made.all()
    exhausted = get_policy(TRADITIONAL)(
        np.zeros(N), make_history(n_hints=depth), unit_views[0], learner_config.support, rng=rng
    )
    assert exhausted.answer_provided.all()


def test_scaffolding_also_withholds_the_answer_until_exhaustion(learner_config, unit_views, rng):
    depth = learner_config.support.max_hint_depth
    outcome = get_policy(AI_SCAFFOLDING)(
        np.zeros(N), make_history(n_hints=depth - 1), unit_views[0], learner_config.support, rng=rng
    )
    assert not outcome.answer_provided.any()


def test_the_two_interactive_ladders_have_equal_depth(learner_config):
    assert max_steps(TRADITIONAL, learner_config.support) == max_steps(
        AI_SCAFFOLDING, learner_config.support
    )


def test_static_ai_is_non_interactive(learner_config):
    assert max_steps(STATIC_AI, learner_config.support) == 1


def test_ladder_support_rises_with_depth(learner_config, unit_views, rng):
    levels = []
    for n_hints in range(learner_config.support.max_hint_depth):
        outcome = get_policy(TRADITIONAL)(
            np.zeros(N),
            make_history(n_hints=n_hints, streak=0),
            unit_views[0],
            dataclasses.replace(learner_config.support, persistence_policy=PERSISTENT),
            rng=rng,
        )
        levels.append(float(outcome.h[0]))
    assert levels == sorted(levels)
    assert levels[-1] > levels[0]


# ---------------------------------------------------------------------------
# Adaptation: the mechanism that distinguishes the conditions
# ---------------------------------------------------------------------------


def test_prewritten_ladders_cannot_diagnose(learner_config, unit_views, rng):
    """`traditional` adaptation is a constant; scaffolding's varies."""
    traditional = get_policy(TRADITIONAL)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    assert np.allclose(traditional.adaptation, learner_config.support.adaptation_traditional)


def test_scaffolding_adaptation_reflects_the_diagnosis(learner_config, unit_views):
    cfg = dataclasses.replace(learner_config.support, diagnosis_accuracy=1.0)
    always = get_policy(AI_SCAFFOLDING)(
        np.zeros(N), make_history(), unit_views[0], cfg, rng=np.random.default_rng(0)
    )
    assert np.allclose(
        always.adaptation, cfg.adaptation_scaffolding_base + cfg.adaptation_scaffolding_gain
    )

    never = get_policy(AI_SCAFFOLDING)(
        np.zeros(N),
        make_history(),
        unit_views[0],
        dataclasses.replace(cfg, diagnosis_accuracy=0.0),
        rng=np.random.default_rng(0),
    )
    assert np.allclose(never.adaptation, cfg.adaptation_scaffolding_base)


def test_scaffolding_adapts_better_than_a_prewritten_ladder(learner_config, unit_views, rng):
    scaffolding = get_policy(AI_SCAFFOLDING)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    traditional = get_policy(TRADITIONAL)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    assert scaffolding.adaptation.mean() > traditional.adaptation.mean()


def test_substitution_adapts_worst_of_all(learner_config, unit_views, rng):
    substitution = get_policy(AI_SUBSTITUTION)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    traditional = get_policy(TRADITIONAL)(
        np.zeros(N), make_history(), unit_views[0], learner_config.support, rng=rng
    )
    assert substitution.adaptation.mean() < traditional.adaptation.mean()


# ---------------------------------------------------------------------------
# Mismatch
# ---------------------------------------------------------------------------


def test_mismatch_is_symmetric_and_bounded(learner_config):
    cfg = learner_config.support
    too_hard = difficulty_mismatch(np.zeros(N), np.full(N, 2.0), cfg)
    too_easy = difficulty_mismatch(np.zeros(N), np.full(N, -2.0), cfg)
    np.testing.assert_allclose(too_hard, too_easy)
    extreme = difficulty_mismatch(np.zeros(N), np.full(N, 100.0), cfg)
    assert np.allclose(extreme, STATE_UPPER)


def test_a_well_matched_unit_has_no_mismatch(learner_config):
    matched = difficulty_mismatch(np.full(N, 0.7), np.full(N, 0.7), learner_config.support)
    assert np.allclose(matched, STATE_LOWER)


# ---------------------------------------------------------------------------
# Persistence policies
# ---------------------------------------------------------------------------


def test_persistent_leaves_support_alone(learner_config):
    cfg = dataclasses.replace(learner_config.support, persistence_policy=PERSISTENT)
    nominal = np.full(N, 0.6)
    np.testing.assert_allclose(apply_persistence(nominal, make_history(streak=5), cfg), nominal)


def test_gradual_fading_decays_geometrically(learner_config):
    cfg = dataclasses.replace(learner_config.support, persistence_policy=GRADUAL_FADING)
    nominal = np.full(N, 0.6)
    faded = [
        float(apply_persistence(nominal, make_history(streak=streak), cfg)[0])
        for streak in range(4)
    ]
    assert faded == sorted(faded, reverse=True)
    assert faded[1] == pytest.approx(faded[0] * cfg.fade_base)


def test_immediate_withdrawal_removes_support_at_the_threshold(learner_config):
    cfg = dataclasses.replace(learner_config.support, persistence_policy=IMMEDIATE_WITHDRAWAL)
    below = apply_persistence(
        np.full(N, 0.6), make_history(streak=cfg.withdrawal_success_threshold - 1), cfg
    )
    at = apply_persistence(
        np.full(N, 0.6), make_history(streak=cfg.withdrawal_success_threshold), cfg
    )
    assert np.allclose(below, 0.6)
    assert np.allclose(at, STATE_LOWER)


@pytest.mark.parametrize("policy", (PERSISTENT, GRADUAL_FADING, IMMEDIATE_WITHDRAWAL))
def test_no_policy_increases_support_with_success(learner_config, policy):
    cfg = dataclasses.replace(learner_config.support, persistence_policy=policy)
    nominal = np.full(N, 0.6)
    levels = [
        float(apply_persistence(nominal, make_history(streak=streak), cfg)[0])
        for streak in range(6)
    ]
    assert all(later <= earlier for earlier, later in zip(levels, levels[1:]))


def test_persistence_is_orthogonal_to_condition(learner_config, unit_views, rng):
    """Any condition can run under any policy."""
    for condition in ALL_CONDITIONS:
        for policy in (PERSISTENT, GRADUAL_FADING, IMMEDIATE_WITHDRAWAL):
            cfg = dataclasses.replace(learner_config.support, persistence_policy=policy)
            outcome = get_policy(condition)(
                np.zeros(N), make_history(streak=3), unit_views[0], cfg, rng=rng
            )
            assert outcome.h.max() <= outcome.h_nominal.max() + 1e-12


# ---------------------------------------------------------------------------
# SupportFaded (eq. 25 input)
# ---------------------------------------------------------------------------


def test_support_faded_is_one_when_nothing_was_consumed():
    faded = support_faded(np.zeros(N), np.full(N, 0.6))
    assert np.allclose(faded, STATE_UPPER)


def test_support_faded_is_zero_when_nothing_was_withdrawn():
    faded = support_faded(np.full(N, 0.6), np.full(N, 0.6))
    assert np.allclose(faded, STATE_LOWER)


def test_support_faded_is_one_when_no_support_was_offered():
    """No support offered at all is fully faded support, not a division by zero."""
    faded = support_faded(np.zeros(N), np.zeros(N))
    assert np.allclose(faded, STATE_UPPER)
    assert np.isfinite(faded).all()


def test_support_faded_is_partial_in_between():
    faded = support_faded(np.full(N, 0.3), np.full(N, 0.6))
    assert np.allclose(faded, 0.5)
