"""Equations (21)-(25), the §7.6 calibration deviation, and clip accounting.

Each equation is tested against the algebra written out in the brief, not against
a reimplementation of itself: the expected values below are computed inline from
the same inputs, so a sign flip inside `updates.py` fails here rather than
propagating a plausible-looking trajectory.

The clip accounting has its own tests because it is the diagnostic that caught a
wrong parameterisation during this build -- `D` was being clipped on 47% of
updates, which is what "heavy clipping means the parameters are wrong" looks like
in practice.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learners.config import STATE_LOWER, STATE_UPPER
from src.learners.updates import (
    ClipCounter,
    accumulate_brier,
    apply_updates,
    calibration_from_brier,
    clip_state,
    decay_only,
    memory_decay_rate,
    update_dependence,
    update_knowledge,
    update_memory,
    update_reasoning,
)
from tests.test_effort import make_proxies

N = 6


@pytest.fixture
def counter() -> ClipCounter:
    return ClipCounter()


# ---------------------------------------------------------------------------
# Equation 21
# ---------------------------------------------------------------------------


def test_knowledge_matches_the_written_algebra(counter):
    knowledge = np.full(N, 0.4)
    alpha = np.full(N, 0.1)
    delta = np.full(N, 0.02)
    effort = np.full(N, 0.8)
    effectiveness = np.full(N, 0.7)
    expected = 0.4 + 0.1 * 0.8 * 0.7 * (1 - 0.4) - 0.02 * 0.4
    result = update_knowledge(knowledge, alpha, delta, effort, effectiveness, counter)
    np.testing.assert_allclose(result, expected)


def test_knowledge_needs_both_effort_and_effectiveness(counter):
    """The gain is a product, so either factor at zero teaches nothing.

    This is the project's hypothesis in miniature: it is the same product
    structure the Phase IV plasticity update uses for E * Z.
    """
    knowledge = np.full(N, 0.4)
    alpha, delta = np.full(N, 0.1), np.zeros(N)
    no_effort = update_knowledge(knowledge, alpha, delta, np.zeros(N), np.ones(N), counter)
    no_effectiveness = update_knowledge(knowledge, alpha, delta, np.ones(N), np.zeros(N), counter)
    np.testing.assert_allclose(no_effort, knowledge)
    np.testing.assert_allclose(no_effectiveness, knowledge)


def test_knowledge_gain_shrinks_as_it_saturates(counter):
    """The (1 - K) headroom term."""
    alpha, delta = np.full(N, 0.2), np.zeros(N)
    effort, effectiveness = np.ones(N), np.ones(N)
    low = update_knowledge(np.full(N, 0.1), alpha, delta, effort, effectiveness, counter)
    high = update_knowledge(np.full(N, 0.9), alpha, delta, effort, effectiveness, counter)
    assert (low - 0.1).mean() > (high - 0.9).mean()


def test_forgetting_is_unconditional(counter):
    """eq. (21)'s loss term applies whether or not the unit was practised."""
    result = update_knowledge(
        np.full(N, 0.5), np.full(N, 0.1), np.full(N, 0.05), np.zeros(N), np.zeros(N), counter
    )
    assert np.all(result < 0.5)


# ---------------------------------------------------------------------------
# Equation 22
# ---------------------------------------------------------------------------


def test_memory_matches_the_written_algebra(learner_config, counter):
    cfg = learner_config.updates
    memory, delta = np.full(N, 0.5), np.full(N, 0.02)
    proxies = make_proxies(retrieval=np.full(N, 0.75), correct_after_error=np.ones(N))
    decay = min(cfg.m_decay_scale * 0.02, 1.0)
    expected = (1 - decay) * 0.5 + cfg.eta_M * 0.75 + cfg.eta_C * 1.0
    np.testing.assert_allclose(update_memory(memory, delta, proxies, cfg, counter), expected)


def test_memory_decay_rate_is_derived_from_delta(learner_config):
    """eq. (16) draws no delta_M, so it is scaled off delta_i. ASSUMPTION."""
    cfg = learner_config.updates
    delta = np.full(N, 0.02)
    np.testing.assert_allclose(memory_decay_rate(delta, cfg), cfg.m_decay_scale * 0.02)


def test_memory_decay_rate_is_clipped_at_one(learner_config):
    """(1 - delta_M) must stay non-negative, or decay flips the trace's sign."""
    rate = memory_decay_rate(np.full(N, 0.9), learner_config.updates)
    assert rate.max() <= STATE_UPPER


def test_retrieval_and_recovery_both_consolidate(learner_config, counter):
    cfg = learner_config.updates
    memory, delta = np.full(N, 0.5), np.full(N, 0.02)
    baseline = update_memory(
        memory, delta, make_proxies(retrieval=np.zeros(N), correct_after_error=np.zeros(N)), cfg, counter
    )
    retrieved = update_memory(
        memory, delta, make_proxies(retrieval=np.ones(N), correct_after_error=np.zeros(N)), cfg, counter
    )
    recovered = update_memory(
        memory, delta, make_proxies(retrieval=np.zeros(N), correct_after_error=np.ones(N)), cfg, counter
    )
    assert np.all(retrieved > baseline)
    assert np.all(recovered > baseline)


# ---------------------------------------------------------------------------
# Equation 23
# ---------------------------------------------------------------------------


def test_reasoning_matches_the_written_algebra(learner_config, counter):
    cfg = learner_config.updates
    proxies = make_proxies(transfer_success=np.ones(N), offloading=np.full(N, 0.25))
    expected = 0.5 + cfg.eta_R * 0.8 * 1.0 - cfg.eta_O * 0.25
    result = update_reasoning(np.full(N, 0.5), np.full(N, 0.8), proxies, cfg, counter)
    np.testing.assert_allclose(result, expected)


def test_reasoning_grows_only_when_effort_transfers(learner_config, counter):
    """The gain is E * TransferSuccess, so effort that does not transfer buys nothing."""
    cfg = learner_config.updates
    reasoning = np.full(N, 0.5)
    no_transfer = update_reasoning(
        reasoning,
        np.ones(N),
        make_proxies(transfer_success=np.zeros(N), offloading=np.zeros(N)),
        cfg,
        counter,
    )
    np.testing.assert_allclose(no_transfer, reasoning)


def test_offloading_erodes_reasoning(learner_config, counter):
    cfg = learner_config.updates
    result = update_reasoning(
        np.full(N, 0.5),
        np.zeros(N),
        make_proxies(transfer_success=np.zeros(N), offloading=np.ones(N)),
        cfg,
        counter,
    )
    assert np.all(result < 0.5)


def test_reasoning_has_no_interior_equilibrium(learner_config, counter):
    """Recorded as a finding: eq. (23) has neither headroom nor decay.

    Iterating a favourable episode drives R to its bound and pins it there. This
    is a property of the equation, not of the rates, and it is why the clip counts
    are surfaced in the run log.
    """
    cfg = learner_config.updates
    reasoning = np.full(N, 0.5)
    proxies = make_proxies(transfer_success=np.ones(N), offloading=np.zeros(N))
    for _ in range(500):
        reasoning = update_reasoning(reasoning, np.ones(N), proxies, cfg, counter)
    assert np.allclose(reasoning, STATE_UPPER)
    assert counter.as_dict()["R"]["clipped"] > 0


# ---------------------------------------------------------------------------
# Equation 25
# ---------------------------------------------------------------------------


def test_dependence_matches_the_written_algebra(learner_config, counter):
    cfg = learner_config.updates
    proxies = make_proxies(
        support_used=np.full(N, 0.6),
        support_faded=np.full(N, 0.5),
        independent_success=np.ones(N),
    )
    expected = 0.4 + cfg.eta_D * 0.6 - cfg.eta_F * 0.5 * 1.0
    np.testing.assert_allclose(
        update_dependence(np.full(N, 0.4), proxies, cfg, counter), expected
    )


def test_consuming_support_raises_dependence(learner_config, counter):
    cfg = learner_config.updates
    result = update_dependence(
        np.full(N, 0.4),
        make_proxies(support_used=np.ones(N), support_faded=np.zeros(N), independent_success=np.zeros(N)),
        cfg,
        counter,
    )
    assert np.all(result > 0.4)


def test_dependence_falls_only_on_withdrawn_support_plus_success(learner_config, counter):
    """The loss term is a product; neither factor alone reduces D."""
    cfg = learner_config.updates
    start = np.full(N, 0.4)
    faded_only = update_dependence(
        start,
        make_proxies(support_used=np.zeros(N), support_faded=np.ones(N), independent_success=np.zeros(N)),
        cfg,
        counter,
    )
    success_only = update_dependence(
        start,
        make_proxies(support_used=np.zeros(N), support_faded=np.zeros(N), independent_success=np.ones(N)),
        cfg,
        counter,
    )
    both = update_dependence(
        start,
        make_proxies(support_used=np.zeros(N), support_faded=np.ones(N), independent_success=np.ones(N)),
        cfg,
        counter,
    )
    np.testing.assert_allclose(faded_only, start)
    np.testing.assert_allclose(success_only, start)
    assert np.all(both < start)


# ---------------------------------------------------------------------------
# Equation 24 -- the §7.6 deviation
# ---------------------------------------------------------------------------


def test_calibration_is_one_minus_the_running_brier(learner_config, counter):
    cfg = learner_config.updates
    brier_sum = np.full(N, 0.5)
    brier_count = np.full(N, 4.0)
    result = calibration_from_brier(brier_sum, brier_count, np.zeros(N), cfg, counter)
    np.testing.assert_allclose(result, 1.0 - (0.5 / 4.0) / cfg.brier_max)


def test_better_calibration_gives_higher_c(learner_config, counter):
    """Brier is a loss, so C inverts it -- the direction every other state has."""
    cfg = learner_config.updates
    good = calibration_from_brier(np.full(N, 0.1), np.full(N, 10.0), np.zeros(N), cfg, counter)
    bad = calibration_from_brier(np.full(N, 5.0), np.full(N, 10.0), np.zeros(N), cfg, counter)
    assert np.all(good > bad)


def test_learners_with_no_forecasts_keep_their_initial_c(learner_config, counter):
    """A learner is not assigned a calibration they have not demonstrated."""
    cfg = learner_config.updates
    fallback = np.full(N, 0.37)
    result = calibration_from_brier(np.zeros(N), np.zeros(N), fallback, cfg, counter)
    np.testing.assert_allclose(result, fallback)


def test_accumulate_brier_excludes_unmade_forecasts():
    """A response that never happened must not enter the denominator."""
    brier_sum, brier_count = np.zeros(2), np.zeros(2)
    confidence = np.array([[0.9, 0.4], [0.9, 0.4]])
    correct = np.array([[1.0, 0.0], [1.0, 0.0]])
    mask = np.array([[True, True], [False, True]])
    total, count = accumulate_brier(brier_sum, brier_count, confidence, correct, mask)
    np.testing.assert_allclose(count, [2.0, 1.0])
    np.testing.assert_allclose(total, [0.01 + 0.16, 0.16])


def test_accumulate_brier_rejects_misaligned_inputs():
    with pytest.raises(ValueError, match="must align"):
        accumulate_brier(
            np.zeros(2), np.zeros(2), np.zeros((2, 2)), np.zeros((2, 3)), np.ones((2, 2), dtype=bool)
        )


# ---------------------------------------------------------------------------
# Clip accounting
# ---------------------------------------------------------------------------


def test_clip_counts_values_that_moved(counter):
    values = np.array([-0.5, 0.5, 1.5])
    result = clip_state("K", values, counter)
    np.testing.assert_allclose(result, [STATE_LOWER, 0.5, STATE_UPPER])
    assert counter.as_dict()["K"] == {"clipped": 2, "total": 3, "share": pytest.approx(2 / 3)}


def test_clip_accumulates_across_calls(counter):
    clip_state("D", np.array([1.5, 0.5]), counter)
    clip_state("D", np.array([1.5, 0.5]), counter)
    assert counter.as_dict()["D"]["clipped"] == 2
    assert counter.as_dict()["D"]["total"] == 4


def test_clip_refuses_non_finite_input(counter):
    """Clipping bounds a value; it does not repair one."""
    with pytest.raises(ValueError, match="non-finite before clipping"):
        clip_state("M", np.array([np.nan, 0.5]), counter)


def test_counter_merge_keeps_both_totals():
    first, second = ClipCounter(), ClipCounter()
    clip_state("K", np.array([1.5]), first)
    clip_state("K", np.array([0.5]), second)
    first.merge(second)
    assert first.as_dict()["K"] == {"clipped": 1, "total": 2, "share": 0.5}


def test_summary_lines_are_readable(counter):
    clip_state("R", np.array([1.5, 0.5]), counter)
    assert any("R: 1 / 2 clipped" in line for line in counter.summary_lines())


# ---------------------------------------------------------------------------
# Orchestration and the retention projection
# ---------------------------------------------------------------------------


def test_apply_updates_advances_every_component(learner_state, learner_config, counter):
    before = {name: getattr(learner_state, name).copy() for name in ("K", "M", "R", "C", "D")}
    n = learner_state.n_learners
    proxies = make_proxies(
        **{
            name: np.full(n, value)
            for name, value in (
                ("attempt", 1.0),
                ("retrieval", 0.75),
                ("explanation", 0.5),
                ("answer_provided", 0.0),
                ("offloading", 0.5),
                ("adaptation", 0.5),
                ("mismatch", 0.3),
                ("coverage", 1.0),
                ("correctness", 1.0),
                ("correct_after_error", 1.0),
                ("transfer_success", 1.0),
                ("support_used", 0.3),
                ("support_faded", 0.5),
                ("independent_success", 1.0),
            )
        }
    )
    apply_updates(
        learner_state,
        proxies,
        np.full(n, 0.7),
        np.full(n, 0.8),
        np.column_stack([np.full(n, 0.6), np.full(n, 0.6)]),
        np.column_stack([np.ones(n, dtype=bool), np.ones(n, dtype=bool)]),
        learner_config.updates,
        counter,
        mask=np.ones((n, 2), dtype=bool),
    )
    assert learner_state.episodes_completed == 1
    assert not np.allclose(learner_state.K, before["K"])
    assert (learner_state.brier_count == 2).all()
    # independent_success was 1 for everyone, so the streak advanced.
    assert (learner_state.independent_success_streak == 1).all()


def test_streak_resets_when_a_learner_does_not_succeed_alone(
    learner_state, learner_config, counter
):
    n = learner_state.n_learners
    learner_state.independent_success_streak[:] = 4
    proxies = make_proxies(
        attempt=np.ones(n),
        retrieval=np.full(n, 0.5),
        explanation=np.full(n, 0.5),
        answer_provided=np.zeros(n),
        offloading=np.full(n, 0.5),
        adaptation=np.full(n, 0.5),
        mismatch=np.full(n, 0.3),
        coverage=np.ones(n),
        correctness=np.ones(n),
        correct_after_error=np.zeros(n),
        transfer_success=np.zeros(n),
        support_used=np.full(n, 0.3),
        support_faded=np.full(n, 0.5),
        independent_success=np.zeros(n),
    )
    apply_updates(
        learner_state,
        proxies,
        np.full(n, 0.5),
        np.full(n, 0.5),
        np.column_stack([np.full(n, 0.5)]),
        np.column_stack([np.zeros(n, dtype=bool)]),
        learner_config.updates,
        counter,
        mask=np.ones((n, 1), dtype=bool),
    )
    assert (learner_state.independent_success_streak == 0).all()


def test_decay_only_does_not_mutate_the_source(learner_state, learner_config, counter):
    before = learner_state.K.copy()
    decay_only(learner_state, 10, learner_config.updates, counter)
    np.testing.assert_allclose(learner_state.K, before)


def test_longer_intervals_retain_less(learner_state, learner_config, counter):
    short = decay_only(learner_state, 2, learner_config.updates, counter)
    long = decay_only(learner_state, 20, learner_config.updates, counter)
    assert long.K.mean() < short.K.mean()
    assert long.M.mean() < short.M.mean()


def test_decay_only_leaves_r_c_and_d_untouched(learner_state, learner_config, counter):
    """eq. (23) and (25) have no decay term, and C is a function of past forecasts."""
    projected = decay_only(learner_state, 10, learner_config.updates, counter)
    np.testing.assert_allclose(projected.R, learner_state.R)
    np.testing.assert_allclose(projected.C, learner_state.C)
    np.testing.assert_allclose(projected.D, learner_state.D)


def test_zero_interval_is_a_no_op(learner_state, learner_config, counter):
    projected = decay_only(learner_state, 0, learner_config.updates, counter)
    np.testing.assert_allclose(projected.K, learner_state.K)


def test_negative_interval_raises(learner_state, learner_config, counter):
    with pytest.raises(ValueError, match="n_episodes must be >= 0"):
        decay_only(learner_state, -1, learner_config.updates, counter)
