"""`config/learners.yaml` loading, the low/medium/high sweep, and the guards.

The rule under test everywhere here: configuration is **not coerced**. An unknown
key, a non-PSD correlation matrix, weights that do not sum to one, or a sign that
would reverse an equation's direction all raise. The run log records a config
hash, so a silently accepted value would produce a silently wrong provenance
record.
"""

from __future__ import annotations

import dataclasses

import pytest
import yaml

from src.learners.config import (
    ALLOWED_SETTINGS,
    REQUIRED_STREAMS,
    STATE_COMPONENTS,
    EffortConfig,
    LearnerConfigError,
    PopulationConfig,
    is_sweep,
    load_learner_config,
    resolve_settings,
)
from tests.conftest import LEARNER_CONFIG_PATH


def test_loads_the_real_config():
    cfg = load_learner_config(LEARNER_CONFIG_PATH)
    assert cfg.parameter_setting == "medium"
    assert cfg.engine.name == "logistic"
    assert cfg.seeds.streams == REQUIRED_STREAMS
    assert len(cfg.population.state_means) == len(STATE_COMPONENTS)


@pytest.mark.parametrize("setting", ALLOWED_SETTINGS)
def test_every_sensitivity_arm_loads(setting):
    """§7.2: the baseline values are assumptions, so all three arms must be valid."""
    cfg = load_learner_config(LEARNER_CONFIG_PATH, setting=setting)
    assert cfg.parameter_setting == setting


def test_settings_produce_different_hashes():
    hashes = {
        setting: load_learner_config(LEARNER_CONFIG_PATH, setting=setting).config_hash()
        for setting in ALLOWED_SETTINGS
    }
    assert len(set(hashes.values())) == len(ALLOWED_SETTINGS)


def test_config_hash_is_stable_across_loads():
    first = load_learner_config(LEARNER_CONFIG_PATH).config_hash()
    second = load_learner_config(LEARNER_CONFIG_PATH).config_hash()
    assert first == second
    assert len(first) == 64


def test_sweep_detection_is_exact_key_equality():
    assert is_sweep({"low": 1, "medium": 2, "high": 3})
    # A missing baseline is a typo, not a two-point sweep.
    assert not is_sweep({"low": 1, "high": 3})
    # A per-stratum table that happens to contain `low` is not a sweep.
    assert not is_sweep({"low": 1, "medium": 2, "high": 3, "extra": 4})


def test_resolve_settings_walks_nested_structures():
    raw = {
        "a": {"low": 1, "medium": 2, "high": 3},
        "b": {"nested": {"low": [1, 2], "medium": [3, 4], "high": [5, 6]}},
        "c": "fixed",
    }
    assert resolve_settings(raw, "high") == {"a": 3, "b": {"nested": [5, 6]}, "c": "fixed"}


def test_resolve_settings_rejects_unknown_setting():
    with pytest.raises(LearnerConfigError, match="parameter_setting must be one of"):
        resolve_settings({"a": 1}, "medium-ish")


def test_unknown_key_raises(tmp_path):
    raw = yaml.safe_load(LEARNER_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["engine"]["temperature"] = 0.7
    path = tmp_path / "learners.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LearnerConfigError, match="unknown key"):
        load_learner_config(path)


def test_missing_key_raises(tmp_path):
    raw = yaml.safe_load(LEARNER_CONFIG_PATH.read_text(encoding="utf-8"))
    del raw["response"]["omega"]
    path = tmp_path / "learners.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(LearnerConfigError, match="missing required key 'omega'"):
        load_learner_config(path)


def test_missing_file_raises():
    with pytest.raises(LearnerConfigError, match="config file not found"):
        load_learner_config(LEARNER_CONFIG_PATH.parent / "no_such_config.yaml")


# ---------------------------------------------------------------------------
# Guards on values the equations depend on
# ---------------------------------------------------------------------------


def test_non_psd_correlation_raises(learner_config):
    population = learner_config.population
    broken = [list(row) for row in population.correlation]
    broken[0][1] = broken[1][0] = 0.99
    broken[0][2] = broken[2][0] = -0.99
    broken[1][2] = broken[2][1] = 0.99
    with pytest.raises(LearnerConfigError, match="positive semi-definite"):
        dataclasses.replace(
            population, correlation=tuple(tuple(row) for row in broken)
        )


def test_asymmetric_correlation_raises(learner_config):
    broken = [list(row) for row in learner_config.population.correlation]
    broken[0][1] = 0.1
    with pytest.raises(LearnerConfigError, match="symmetric"):
        dataclasses.replace(
            learner_config.population, correlation=tuple(tuple(row) for row in broken)
        )


def test_stratum_weights_must_sum_to_one(learner_config):
    with pytest.raises(LearnerConfigError, match="must sum to 1"):
        dataclasses.replace(learner_config.population, stratum_weights=(0.3, 0.3, 0.3))


def test_stratum_list_lengths_must_match(learner_config):
    with pytest.raises(LearnerConfigError, match="read positionally"):
        dataclasses.replace(learner_config.population, stratum_k_shift=(0.0, 0.1))


def test_negative_b_slope_raises(learner_config):
    """The difficulty -> b_u map must be monotone increasing."""
    with pytest.raises(LearnerConfigError, match="monotone increasing"):
        dataclasses.replace(learner_config.curriculum, b_slope=-0.6)


def test_non_positive_omega_raises(learner_config):
    """eq. (18) adds omega*h; a non-positive omega is a modelling claim."""
    with pytest.raises(LearnerConfigError, match="omega must be > 0"):
        dataclasses.replace(learner_config.response, omega=0.0)


def test_negative_update_rate_raises(learner_config):
    """A negative eta would silently reverse an equation's direction."""
    with pytest.raises(LearnerConfigError, match="silently reverse the direction"):
        dataclasses.replace(learner_config.updates, eta_R=-0.01)


def test_decreasing_hint_ladder_raises(learner_config):
    with pytest.raises(LearnerConfigError, match="non-decreasing"):
        dataclasses.replace(learner_config.support, hint_ladder_h=(0.6, 0.3, 0.1))


def test_ladders_must_have_equal_depth(learner_config):
    """Otherwise a condition contrast is also a ladder-length contrast."""
    with pytest.raises(LearnerConfigError, match="same depth"):
        dataclasses.replace(learner_config.support, scaffolding_h=(0.2, 0.4))


def test_zero_probe_h_raises(learner_config):
    """eq. (26) is identically zero at h = 0."""
    with pytest.raises(LearnerConfigError, match="identically zero at h = 0"):
        dataclasses.replace(learner_config.checkpoints, probe_h=0.0)


def test_zero_retention_interval_raises(learner_config):
    with pytest.raises(LearnerConfigError, match="zero interval measures immediate accuracy"):
        dataclasses.replace(learner_config.checkpoints, retention_interval_episodes=0)


def test_reordered_seed_streams_raise(learner_config):
    """Stream order fixes which spawned child each consumer gets."""
    with pytest.raises(LearnerConfigError, match="stream order fixes"):
        dataclasses.replace(learner_config.seeds, streams=("responses", "population", "tutor", "checkpoints"))


def test_unknown_condition_raises(learner_config):
    with pytest.raises(LearnerConfigError, match="unknown condition"):
        dataclasses.replace(learner_config.run, conditions=("traditional", "ai_tutor"))


def test_unknown_engine_raises(tmp_path):
    with pytest.raises(LearnerConfigError, match="engine.name must be one of"):
        load_learner_config(LEARNER_CONFIG_PATH, engine="centaur")


def test_cli_overrides_are_validated(learner_config):
    cfg = load_learner_config(LEARNER_CONFIG_PATH, n_learners=7, episodes=3, conditions=["traditional"])
    assert cfg.population.n_learners == 7
    assert cfg.run.episodes == 3
    assert cfg.run.conditions == ("traditional",)


# ---------------------------------------------------------------------------
# The §10.3 zero-effort control
# ---------------------------------------------------------------------------


def test_zero_effort_override_zeroes_every_named_coefficient(learner_config):
    zeroed = learner_config.effort.zeroed()
    for name, value in learner_config.effort.zero_effort_override:
        assert getattr(zeroed, name) == value
    # a0 is not in the override, so it must be untouched.
    assert zeroed.a0 == learner_config.effort.a0


def test_zero_effort_override_covers_a4(learner_config):
    """a4 must be zeroed too, or `AnswerProvided` still separates the conditions.

    The task spec writes the control as `a1 = a2 = a3 = 0`. With `a4 != 0` the
    substitution condition still differs from the others through
    `AnswerProvided`, so effort-mediated differences would not vanish and the
    control would not be a control. Recorded as a deviation in
    `docs/phase3_assumptions.md`.
    """
    named = dict(learner_config.effort.zero_effort_override)
    assert named == {"a1": 0.0, "a2": 0.0, "a3": 0.0, "a4": 0.0}


def test_zero_effort_override_rejects_unknown_coefficient():
    with pytest.raises(LearnerConfigError, match="may only name eq. \\(19\\) coefficients"):
        EffortConfig.from_dict(
            {
                "a0": 0.0,
                "a1": 1.0,
                "a2": 1.0,
                "a3": 1.0,
                "a4": 1.0,
                "zero_effort_override": {"b1": 0.0},
            }
        )


def test_as_dict_round_trips_through_json():
    import json

    cfg = load_learner_config(LEARNER_CONFIG_PATH)
    payload = json.dumps(cfg.as_dict(), sort_keys=True)
    assert json.loads(payload)["population"]["stratum_weights"] == [0.30, 0.50, 0.20]


def test_population_config_rejects_wrong_state_keys():
    with pytest.raises(LearnerConfigError, match="exactly the keys"):
        PopulationConfig.from_dict(
            {
                "n_learners": 10,
                "stratum_names": ["a"],
                "stratum_weights": [1.0],
                "state_means": {"K": 0.5},
                "state_sds": {"K": 0.1},
                "stratum_k_shift": [0.0],
                "stratum_m_shift": [0.0],
                "correlation": [[1.0]],
                "mu_alpha": -2.3,
                "sigma_alpha": 0.4,
                "a_delta": 2.0,
                "b_delta": 80.0,
                "theta_intercept": 0.0,
                "theta_slope": 4.0,
                "theta_center": 0.5,
                "confidence_bias_mu": 0.0,
                "confidence_bias_sd": 0.1,
                "speed_sigma": 0.25,
            }
        )
