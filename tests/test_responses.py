"""Equations (17)-(18), the confidence model, and the non-finite guards.

Two properties matter most here. First, every term's *direction* must be the one
`docs/learner_equations.md` claims -- a term whose sign is wrong produces a
plausible simulation of the opposite hypothesis. Second, a non-finite probability
must raise: `np.nan_to_num` would turn a broken parameterisation into a
population answering at chance, which is indistinguishable from a real result.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learners.config import STATE_LOWER, STATE_UPPER
from src.learners.engine import (
    STAGE_FIRST,
    STAGE_SUPPORTED,
    EpisodeBatch,
    LogisticEngine,
    MinitaurEngine,
    ResponseEngine,
    build_engine,
)
from src.learners.responses import (
    brier_score,
    draw_confidence,
    draw_correct,
    latency_proxy,
    logistic,
    probability_correct,
    response_logit,
    support_request_probability,
)

N = 6


def _args(**overrides):
    base = {
        "theta": np.zeros(N),
        "b_u": np.zeros(N),
        "reasoning": np.full(N, 0.5),
        "memory": np.full(N, 0.5),
        "support_level": np.zeros(N),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------


def test_logistic_is_centred_and_monotone():
    assert logistic(np.array([0.0]))[0] == pytest.approx(0.5)
    values = logistic(np.array([-4.0, -1.0, 0.0, 1.0, 4.0]))
    assert np.all(np.diff(values) > 0)


def test_logistic_does_not_overflow_at_extremes():
    values = logistic(np.array([-1e4, 1e4]))
    assert np.isfinite(values).all()
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)


def test_ability_raises_p_correct(learner_config):
    low = probability_correct(**_args(theta=np.full(N, -1.0)), cfg=learner_config.response)
    high = probability_correct(**_args(theta=np.full(N, 1.0)), cfg=learner_config.response)
    assert np.all(high > low)


def test_difficulty_lowers_p_correct(learner_config):
    easy = probability_correct(**_args(b_u=np.full(N, -1.0)), cfg=learner_config.response)
    hard = probability_correct(**_args(b_u=np.full(N, 1.0)), cfg=learner_config.response)
    assert np.all(hard < easy)


def test_reasoning_and_memory_raise_p_correct(learner_config):
    baseline = probability_correct(**_args(), cfg=learner_config.response)
    more_r = probability_correct(**_args(reasoning=np.full(N, 0.9)), cfg=learner_config.response)
    more_m = probability_correct(**_args(memory=np.full(N, 0.9)), cfg=learner_config.response)
    assert np.all(more_r > baseline)
    assert np.all(more_m > baseline)


def test_support_raises_p_correct(learner_config):
    """eq. (18) is eq. (17) plus omega*h, and omega > 0 is enforced."""
    unaided = probability_correct(**_args(), cfg=learner_config.response)
    supported = probability_correct(
        **_args(support_level=np.full(N, 0.8)), cfg=learner_config.response
    )
    assert np.all(supported > unaided)


def test_eq_18_reduces_to_eq_17_at_zero_support(learner_config):
    """The two equations are one implementation, so this must be exact."""
    with_zero = response_logit(**_args(support_level=np.zeros(N)), cfg=learner_config.response)
    without = (
        np.zeros(N)
        - np.zeros(N)
        + learner_config.response.rho * np.full(N, 0.5)
        + learner_config.response.kappa * np.full(N, 0.5)
    )
    np.testing.assert_allclose(with_zero, without)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def test_nan_input_raises_and_names_the_offender(learner_config):
    theta = np.zeros(N)
    theta[2] = np.nan
    with pytest.raises(ValueError) as error:
        probability_correct(**_args(theta=theta), cfg=learner_config.response)
    message = str(error.value)
    assert "non-finite" in message
    assert "theta" in message
    assert "nan_to_num" in message


def test_infinite_input_raises(learner_config):
    b_u = np.zeros(N)
    b_u[0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        probability_correct(**_args(b_u=b_u), cfg=learner_config.response)


def test_draw_correct_rejects_out_of_range_probabilities():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match=r"P\(correct\) must lie in"):
        draw_correct(np.array([1.5]), rng)


def test_brier_score_of_no_forecasts_raises():
    with pytest.raises(ValueError, match="undefined"):
        brier_score([], [])


def test_brier_score_matches_eq_27_by_hand():
    # (0.9-1)^2 + (0.2-0)^2 = 0.01 + 0.04, mean 0.025
    assert brier_score([0.9, 0.2], [True, False]) == pytest.approx(0.025)


def test_brier_score_direction():
    """Lower is better: the one metric whose direction is inverted."""
    good = brier_score([0.95, 0.05], [True, False])
    bad = brier_score([0.05, 0.95], [True, False])
    assert good < bad


# ---------------------------------------------------------------------------
# Confidence, requests, latency
# ---------------------------------------------------------------------------


def test_confidence_stays_in_bounds_and_tracks_p(learner_config):
    rng = np.random.default_rng(1)
    p = np.linspace(0.0, 1.0, 2000)
    confidence = draw_confidence(p, np.zeros_like(p), learner_config.response, rng)
    assert confidence.min() >= STATE_LOWER
    assert confidence.max() <= STATE_UPPER
    assert np.corrcoef(p, confidence)[0, 1] > 0.9


def test_confidence_noise_makes_brier_non_degenerate(learner_config):
    """Without noise, the Brier score would be a deterministic function of p."""
    rng = np.random.default_rng(2)
    p = np.full(500, 0.7)
    confidence = draw_confidence(p, np.zeros_like(p), learner_config.response, rng)
    assert confidence.std() > 0


def test_confidence_bias_shifts_reports(learner_config):
    rng = np.random.default_rng(3)
    p = np.full(2000, 0.5)
    overconfident = draw_confidence(p, np.full(2000, 0.2), learner_config.response, rng)
    underconfident = draw_confidence(p, np.full(2000, -0.2), learner_config.response, rng)
    assert overconfident.mean() > underconfident.mean()


def test_support_request_rises_with_dependence(learner_config):
    low = support_request_probability(
        np.zeros(N), np.zeros(N), np.full(N, 0.1), learner_config.response
    )
    high = support_request_probability(
        np.zeros(N), np.zeros(N), np.full(N, 0.9), learner_config.response
    )
    assert np.all(high > low)


def test_support_request_falls_with_relative_ability(learner_config):
    weak = support_request_probability(
        np.full(N, -1.0), np.zeros(N), np.full(N, 0.5), learner_config.response
    )
    strong = support_request_probability(
        np.full(N, 1.0), np.zeros(N), np.full(N, 0.5), learner_config.response
    )
    assert np.all(strong < weak)


def test_latency_rises_with_hints_and_attempts(learner_config):
    speed = np.ones(N)
    base = latency_proxy(speed, np.zeros(N), np.zeros(N), learner_config.response)
    hinted = latency_proxy(speed, np.full(N, 3.0), np.zeros(N), learner_config.response)
    attempted = latency_proxy(speed, np.zeros(N), np.ones(N), learner_config.response)
    assert np.all(hinted > base)
    assert np.all(attempted > base)


# ---------------------------------------------------------------------------
# The engine seam
# ---------------------------------------------------------------------------


def _batch(stage=STAGE_FIRST, support=None):
    return EpisodeBatch(
        unit_id="SYNTH_seam_001",
        episode=0,
        condition="traditional",
        stage=stage,
        b_u=np.zeros(N),
        theta=np.zeros(N),
        memory=np.full(N, 0.5),
        reasoning=np.full(N, 0.5),
        dependence=np.full(N, 0.4),
        support_level=np.zeros(N) if support is None else support,
        confidence_bias=np.zeros(N),
    )


def test_logistic_engine_satisfies_the_protocol(learner_config, learner_streams):
    engine = build_engine(learner_config, learner_streams)
    assert isinstance(engine, ResponseEngine)
    assert engine.name == "logistic"


def test_engine_returns_aligned_arrays(learner_config, learner_streams):
    outcome = build_engine(learner_config, learner_streams).respond(_batch())
    assert outcome.n_learners == N
    assert outcome.answer_correct.dtype == bool
    assert outcome.support_requested.dtype == bool
    assert outcome.confidence.min() >= STATE_LOWER
    assert outcome.p_correct.max() <= STATE_UPPER


def test_unaided_stage_refuses_support():
    """§3.1's order of operations is enforced, not merely documented."""
    with pytest.raises(ValueError, match="unaided by definition"):
        _batch(stage=STAGE_FIRST, support=np.full(N, 0.5))


def test_supported_stage_accepts_support():
    batch = _batch(stage=STAGE_SUPPORTED, support=np.full(N, 0.5))
    assert batch.support_level.max() == pytest.approx(0.5)


def test_out_of_range_support_raises():
    with pytest.raises(ValueError, match="support_level h must lie in"):
        _batch(stage=STAGE_SUPPORTED, support=np.full(N, 1.5))


def test_ragged_batch_raises():
    with pytest.raises(ValueError, match="same shape"):
        EpisodeBatch(
            unit_id="u",
            episode=0,
            condition="traditional",
            stage=STAGE_FIRST,
            b_u=np.zeros(N),
            theta=np.zeros(N + 1),
            memory=np.zeros(N),
            reasoning=np.zeros(N),
            dependence=np.zeros(N),
            support_level=np.zeros(N),
            confidence_bias=np.zeros(N),
        )


def test_unknown_stage_raises():
    with pytest.raises(ValueError, match="stage must be one of"):
        _batch(stage="warmup")


def test_minitaur_engine_is_declared_but_not_implemented(learner_config, learner_streams):
    engine = MinitaurEngine(learner_config, learner_streams)
    assert engine.name == "minitaur"
    with pytest.raises(NotImplementedError) as error:
        engine.respond(_batch())
    message = str(error.value)
    assert "reports/minitaur_feasibility.md" in message
    assert "benchmark_minitaur.py" in message


def test_the_report_the_minitaur_error_points_at_exists():
    from tests.conftest import PROJECT_ROOT

    assert (PROJECT_ROOT / "reports" / "minitaur_feasibility.md").exists()


def test_engine_swap_is_one_config_line(learner_config, learner_streams):
    """§10.2: swapping engines must change one config line and nothing else."""
    import dataclasses

    from src.learners.config import EngineConfig

    swapped = dataclasses.replace(learner_config, engine=EngineConfig(name="minitaur"))
    assert isinstance(build_engine(swapped, learner_streams), MinitaurEngine)
    assert isinstance(build_engine(learner_config, learner_streams), LogisticEngine)


def test_checkpoint_engine_draws_from_a_different_stream(learner_config, learner_streams):
    from src.learners.engine import CHECKPOINT_STREAM

    episode_engine = build_engine(learner_config, learner_streams)
    probe_engine = build_engine(learner_config, learner_streams, stream=CHECKPOINT_STREAM)
    first = episode_engine.respond(_batch()).p_correct
    second = probe_engine.respond(_batch()).p_correct
    # p_correct is deterministic given the batch, so compare the stochastic part.
    assert not np.allclose(
        episode_engine.respond(_batch()).confidence, probe_engine.respond(_batch()).confidence
    )
    np.testing.assert_allclose(first, second)
