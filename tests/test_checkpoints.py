"""Checkpoint assessments with all support removed (§7.7).

The property this file exists for is the non-mutation guarantee: a checkpoint
that advanced the state would make the act of measuring change the trajectory,
and every later episode would carry the measurement's own effect. It is asserted
here directly, on every array of the state.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learners.checkpoints import (
    CHECKPOINT_COLUMNS,
    condition_separation,
    expected_calibration_error,
    run_checkpoint,
    stratum_monotonicity,
    support_gap,
)
from src.learners.config import AI_SCAFFOLDING, AI_SUBSTITUTION, STATE_LOWER, STATE_UPPER, TRADITIONAL
from src.learners.engine import CHECKPOINT_STREAM, build_engine
from src.learners.population import LearnerState
from src.learners.updates import ClipCounter


def run_one(state, units, condition, learner_config, learner_streams, episode=9):
    engine = build_engine(learner_config, learner_streams, stream=CHECKPOINT_STREAM)
    rng = learner_streams.generator(CHECKPOINT_STREAM)
    return run_checkpoint(
        state,
        units,
        condition,
        engine,
        learner_config,
        rng,
        episode=episode,
        counter=ClipCounter(),
    )


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------


def test_a_checkpoint_does_not_mutate_learner_state(
    learner_state, unit_views, learner_config, learner_streams
):
    before = {
        name: getattr(learner_state, name).copy() for name in LearnerState._ARRAY_FIELDS
    }
    episodes_before = learner_state.episodes_completed

    run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)

    for name, original in before.items():
        np.testing.assert_array_equal(
            getattr(learner_state, name), original, err_msg=f"checkpoint mutated {name}"
        )
    assert learner_state.episodes_completed == episodes_before


def test_repeated_checkpoints_leave_the_state_identical(
    learner_state, unit_views, learner_config, learner_streams
):
    before = learner_state.K.copy()
    for _ in range(3):
        run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    np.testing.assert_array_equal(learner_state.K, before)


# ---------------------------------------------------------------------------
# Shape and content
# ---------------------------------------------------------------------------


def test_one_row_per_learner_with_the_declared_columns(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    assert list(frame.columns) == list(CHECKPOINT_COLUMNS)
    assert len(frame) == learner_state.n_learners
    assert (frame["condition"] == TRADITIONAL).all()
    assert (frame["checkpoint_episode"] == 9).all()


def test_every_accuracy_is_a_proportion(learner_state, unit_views, learner_config, learner_streams):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    for column in (
        "unaided_accuracy_trained",
        "supported_accuracy_trained",
        "near_transfer_accuracy",
        "far_transfer_accuracy",
        "retention_accuracy",
        "brier_score",
        "expected_calibration_error",
        "dependence",
    ):
        values = frame[column].to_numpy()
        assert values.min() >= STATE_LOWER, column
        assert values.max() <= STATE_UPPER, column


def test_item_counts_match_the_configuration(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    cfg = learner_config.checkpoints
    assert (frame["n_trained_items"] == cfg.n_trained_items).all()
    assert (frame["n_near_items"] == cfg.n_near_items).all()
    assert (frame["n_far_items"] == cfg.n_far_items).all()
    assert (frame["retention_interval_episodes"] == cfg.retention_interval_episodes).all()


def test_transfer_probes_are_harder_than_trained_items(
    learner_state, unit_views, learner_config, learner_streams
):
    """near_b_delta and far_b_delta both shift b_u upward."""
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    assert frame["near_transfer_accuracy"].mean() < frame["unaided_accuracy_trained"].mean()
    assert frame["far_transfer_accuracy"].mean() < frame["near_transfer_accuracy"].mean()


def test_retention_is_below_immediate_accuracy(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    assert frame["retention_accuracy"].mean() < frame["unaided_accuracy_trained"].mean()


def test_a_checkpoint_with_no_trained_items_raises(
    learner_state, learner_config, learner_streams
):
    """A checkpoint at episode 0 is a specification error, not an empty result."""
    with pytest.raises(ValueError, match="no units have been taught yet"):
        run_one(learner_state, (), TRADITIONAL, learner_config, learner_streams)


def test_a_short_corpus_probes_with_replacement(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views[:2], TRADITIONAL, learner_config, learner_streams)
    assert (frame["n_trained_items"] == learner_config.checkpoints.n_trained_items).all()


# ---------------------------------------------------------------------------
# Equation 26
# ---------------------------------------------------------------------------


def test_support_gap_is_supported_minus_unaided():
    np.testing.assert_allclose(
        support_gap(np.array([0.8, 0.5]), np.array([0.6, 0.5])), [0.2, 0.0]
    )


def test_support_gap_is_positive_at_the_defaults(
    learner_state, unit_views, learner_config, learner_streams
):
    """eq. (18) adds omega*h, so a positive gap is expected in every condition."""
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    assert frame["support_gap"].mean() > 0
    np.testing.assert_allclose(
        frame["support_gap"].to_numpy(),
        frame["supported_accuracy_trained"].to_numpy()
        - frame["unaided_accuracy_trained"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# Expected calibration error
# ---------------------------------------------------------------------------


def test_ece_is_zero_for_a_perfectly_calibrated_forecaster():
    confidence = np.array([[1.0, 1.0, 0.0, 0.0]])
    correct = np.array([[1.0, 1.0, 0.0, 0.0]])
    assert expected_calibration_error(confidence, correct, 10)[0] == pytest.approx(0.0)


def test_ece_is_one_for_a_confidently_wrong_forecaster():
    confidence = np.array([[1.0, 1.0]])
    correct = np.array([[0.0, 0.0]])
    assert expected_calibration_error(confidence, correct, 10)[0] == pytest.approx(1.0)


def test_ece_puts_confidence_of_one_in_the_last_bin():
    """Otherwise the bin index runs one past the end of the array."""
    values = expected_calibration_error(np.array([[1.0]]), np.array([[1.0]]), 5)
    assert np.isfinite(values).all()


def test_ece_is_computed_per_learner():
    confidence = np.array([[0.9, 0.9], [0.5, 0.5]])
    correct = np.array([[1.0, 1.0], [1.0, 1.0]])
    values = expected_calibration_error(confidence, correct, 10)
    assert values.shape == (2,)
    assert values[0] < values[1]


def test_ece_rejects_zero_bins():
    with pytest.raises(ValueError, match="n_bins must be >= 1"):
        expected_calibration_error(np.array([[0.5]]), np.array([[1.0]]), 0)


def test_ece_of_no_forecasts_raises():
    with pytest.raises(ValueError, match="undefined"):
        expected_calibration_error(np.zeros((2, 0)), np.zeros((2, 0)), 10)


# ---------------------------------------------------------------------------
# §10.6 condition separation
# ---------------------------------------------------------------------------


def test_separation_is_not_assessed_when_an_arm_is_missing(
    learner_state, unit_views, learner_config, learner_streams
):
    """Reporting `not separated` for an absent arm would fabricate a signal."""
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    assert condition_separation(frame, learner_config) == ()


def test_separation_is_reported_for_both_outcomes(learner_config):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "condition": [AI_SCAFFOLDING] * 2 + [AI_SUBSTITUTION] * 2,
            "checkpoint_episode": [9, 19] * 2,
            "unaided_accuracy_trained": [0.80, 0.85, 0.40, 0.849],
        }
    )
    verdicts = condition_separation(frame, learner_config)
    assert len(verdicts) == 2
    assert verdicts[0].separated is True
    assert verdicts[1].separated is False
    assert "falsification signal" in verdicts[1].as_line()


def test_separation_verdict_records_the_arithmetic(learner_config):
    import pandas as pd

    frame = pd.DataFrame(
        {
            "condition": [AI_SCAFFOLDING, AI_SUBSTITUTION],
            "checkpoint_episode": [9, 9],
            "unaided_accuracy_trained": [0.8, 0.5],
        }
    )
    verdict = condition_separation(frame, learner_config)[0]
    assert verdict.mean_a == pytest.approx(0.8)
    assert verdict.mean_b == pytest.approx(0.5)
    assert verdict.difference == pytest.approx(0.3)
    assert verdict.threshold == learner_config.checkpoints.separation_threshold


def test_separation_on_an_empty_table_is_empty(learner_config):
    import pandas as pd

    assert condition_separation(pd.DataFrame(), learner_config) == ()


def test_stratum_monotonicity_reads_the_expected_column(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    means = stratum_monotonicity(frame, "unaided_accuracy_trained")
    assert set(means) == set(np.unique(learner_state.stratum_index).tolist())


def test_stratum_monotonicity_rejects_an_unknown_column(
    learner_state, unit_views, learner_config, learner_streams
):
    frame = run_one(learner_state, unit_views, TRADITIONAL, learner_config, learner_streams)
    with pytest.raises(ValueError, match="no column"):
        stratum_monotonicity(frame, "not_a_column")
