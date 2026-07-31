"""The §10.2 validation checks, as executable tests.

These are the brief's own sanity checks on the learner model. Each one asserts a
property the model must have for any of its output to mean anything, and each
fails loudly if violated -- with one deliberate exception, documented on the test
itself: §10.6 condition separation is *recorded* rather than silently passed,
because a run where scaffolding and substitution are indistinguishable after
support removal is a falsification signal about the model, not a test failure to
be fixed by moving a threshold.

Every test here runs a real simulation. They are the slowest tests in the suite
and that is the point: a property that holds for the equations in isolation but
not for a run of them is not a property the study can rely on.
"""

from __future__ import annotations

import dataclasses
import warnings
from pathlib import Path

import numpy as np
import pytest

from src.learners.checkpoints import condition_separation
from src.learners.config import (
    AI_SCAFFOLDING,
    AI_SUBSTITUTION,
    TRADITIONAL,
)
from src.learners.curriculum import load_curriculum
from src.learners.simulate import run_simulation, sha256_file
from src.learners.support import GRADUAL_FADING, PERSISTENT
from tests.conftest import make_units_frame

CONDITIONS = (TRADITIONAL, AI_SCAFFOLDING, AI_SUBSTITUTION)


def simulate(cfg, units, tmp_path: Path, name: str = "run"):
    """Run a simulation into a temporary directory, warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return run_simulation(
            cfg,
            units,
            out_dir=tmp_path / name / "outputs",
            data_dir=tmp_path / name / "data",
            timestamp="FIXED",
        )


def units_at_difficulty(difficulty: int, cfg, tmp_path: Path, name: str):
    """A units table where every unit sits at one difficulty level."""
    frame = make_units_frame(12)
    frame["difficulty"] = difficulty
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return load_curriculum(path, cfg.curriculum)


@pytest.fixture(scope="module")
def validation_run(tmp_path_factory):
    """One shared simulation across every condition, reused by several checks."""
    from src.learners.config import load_learner_config
    from tests.conftest import LEARNER_CONFIG_PATH

    tmp_path = tmp_path_factory.mktemp("validation")
    cfg = load_learner_config(
        LEARNER_CONFIG_PATH, n_learners=600, episodes=40, conditions=list(CONDITIONS)
    )
    frame = make_units_frame(12)
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    units = load_curriculum(path, cfg.curriculum)
    return cfg, simulate(cfg, units, tmp_path, "shared")


# ---------------------------------------------------------------------------
# 1. Monotonicity in prior knowledge
# ---------------------------------------------------------------------------


def test_1_accuracy_is_monotone_in_prior_knowledge(validation_run):
    """Mean baseline accuracy is non-decreasing across the low/medium/high strata."""
    _, result = validation_run
    checkpoints = result.checkpoints
    first = checkpoints[checkpoints["checkpoint_episode"] == checkpoints["checkpoint_episode"].min()]
    means = (
        first.groupby("stratum_index")["unaided_accuracy_trained"].mean().sort_index().to_numpy()
    )
    assert len(means) == 3
    assert np.all(np.diff(means) >= 0), f"accuracy is not monotone across strata: {means}"


# ---------------------------------------------------------------------------
# 2. Difficulty
# ---------------------------------------------------------------------------


def test_2_higher_difficulty_reduces_unaided_accuracy(learner_config, tmp_path):
    """A harder unit means a larger b_u, and b_u enters eq. (17) negatively."""
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, conditions=(TRADITIONAL,)),
    )
    easy = simulate(cfg, units_at_difficulty(1, cfg, tmp_path, "easy"), tmp_path, "easy")
    hard = simulate(cfg, units_at_difficulty(5, cfg, tmp_path, "hard"), tmp_path, "hard")

    easy_accuracy = easy.diagnostics["mean_unaided_correct"].mean()
    hard_accuracy = hard.diagnostics["mean_unaided_correct"].mean()
    assert hard_accuracy < easy_accuracy, (
        f"difficulty 5 gave {hard_accuracy:.4f} and difficulty 1 gave {easy_accuracy:.4f}"
    )


# ---------------------------------------------------------------------------
# 3. Support
# ---------------------------------------------------------------------------


def test_3_support_raises_immediate_accuracy(validation_run):
    """eq. (26) on the same items: supported minus unaided must be positive.

    Measured on the *same* probe items rather than by comparing the episode's
    first and second responses, which would be confounded -- only learners who
    failed the first response ever receive a second one.
    """
    _, result = validation_run
    for condition in CONDITIONS:
        window = result.checkpoints[result.checkpoints["condition"] == condition]
        gap = window["support_gap"].mean()
        assert gap > 0, f"{condition}: support gap is {gap:.4f}"


# ---------------------------------------------------------------------------
# 4. Forgetting
# ---------------------------------------------------------------------------


def test_4_longer_no_practice_intervals_reduce_retention(learner_config, tmp_path):
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=20, conditions=(TRADITIONAL,)),
    )
    units = units_at_difficulty(3, cfg, tmp_path, "units")

    short = dataclasses.replace(
        cfg, checkpoints=dataclasses.replace(cfg.checkpoints, retention_interval_episodes=1)
    )
    long = dataclasses.replace(
        cfg, checkpoints=dataclasses.replace(cfg.checkpoints, retention_interval_episodes=40)
    )
    short_result = simulate(short, units, tmp_path, "short")
    long_result = simulate(long, units, tmp_path, "long")

    short_retention = short_result.checkpoints["retention_accuracy"].mean()
    long_retention = long_result.checkpoints["retention_accuracy"].mean()
    assert long_retention < short_retention, (
        f"40 episodes of no practice gave {long_retention:.4f}, 1 episode gave "
        f"{short_retention:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Fading
# ---------------------------------------------------------------------------


def test_5_fading_plus_independent_success_reduces_dependence(learner_config, tmp_path):
    """eq. (25)'s loss term is SupportFaded * IndependentSuccess.

    Under a fading policy a learner who keeps succeeding alone sees support
    withdrawn, so both factors are non-zero and `D` falls. Under `persistent` the
    support level does not fall, so the loss term is weaker and `D` ends higher.
    """
    base = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=30, conditions=(TRADITIONAL,)),
    )
    units = units_at_difficulty(1, base, tmp_path, "easy_units")

    faded = dataclasses.replace(
        base, support=dataclasses.replace(base.support, persistence_policy=GRADUAL_FADING)
    )
    persistent = dataclasses.replace(
        base, support=dataclasses.replace(base.support, persistence_policy=PERSISTENT)
    )
    faded_result = simulate(faded, units, tmp_path, "faded")
    persistent_result = simulate(persistent, units, tmp_path, "persistent")

    faded_d = faded_result.checkpoints["D"].mean()
    persistent_d = persistent_result.checkpoints["D"].mean()
    assert faded_d < persistent_d, (
        f"fading gave D={faded_d:.4f}, persistent gave D={persistent_d:.4f}"
    )


def test_5b_dependence_falls_over_episodes_under_fading(learner_config, tmp_path):
    """The same property inside one run: D must trend down, not merely differ."""
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=40, conditions=(TRADITIONAL,)),
        support=dataclasses.replace(learner_config.support, persistence_policy=GRADUAL_FADING),
    )
    result = simulate(cfg, units_at_difficulty(1, cfg, tmp_path, "easy2"), tmp_path, "trend")
    by_episode = result.checkpoints.groupby("checkpoint_episode")["D"].mean()
    assert by_episode.is_monotonic_decreasing, by_episode.to_dict()


# ---------------------------------------------------------------------------
# 6. Zero-effort negative control (§10.3)
# ---------------------------------------------------------------------------


def test_6_zero_effort_control_removes_effort_differences(learner_config, tmp_path):
    """With the eq. (19) coefficients zeroed, E cannot differ between conditions.

    Note the control zeroes `a4` as well as `a1..a3`. The task spec writes it as
    `a1 = a2 = a3 = 0`, but with `a4 != 0` the substitution condition still
    differs through `AnswerProvided`, so effort-mediated differences would not
    vanish and the control would not be a control. `test_effort.py` demonstrates
    that directly; recorded in `docs/phase3_assumptions.md`.
    """
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=20, conditions=CONDITIONS),
    )
    units = units_at_difficulty(3, cfg, tmp_path, "control_units")

    baseline = simulate(cfg, units, tmp_path, "baseline")
    control = simulate(
        dataclasses.replace(cfg, effort=cfg.effort.zeroed()), units, tmp_path, "control"
    )

    baseline_effort = baseline.diagnostics.groupby("condition")["mean_effort"].mean()
    control_effort = control.diagnostics.groupby("condition")["mean_effort"].mean()

    assert baseline_effort.max() - baseline_effort.min() > 0.1, (
        f"the baseline shows no effort difference to remove: {baseline_effort.to_dict()}"
    )
    spread = control_effort.max() - control_effort.min()
    assert spread < 1e-12, f"effort still differs under the control: {control_effort.to_dict()}"


# ---------------------------------------------------------------------------
# 7. Determinism (§10.1)
# ---------------------------------------------------------------------------


def test_7_same_seed_gives_a_bitwise_identical_parquet(learner_config, tmp_path):
    """sha256 via `hashlib`, never Python's `hash()`, which is salted per process."""
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=15, conditions=CONDITIONS),
    )
    units = units_at_difficulty(3, cfg, tmp_path, "determinism_units")

    first = simulate(cfg, units, tmp_path, "run_a")
    second = simulate(cfg, units, tmp_path, "run_b")

    first_hash = sha256_file(first.paths.learner_state_parquet)
    second_hash = sha256_file(second.paths.learner_state_parquet)
    assert first_hash == second_hash, f"{first_hash} != {second_hash}"
    assert first.config_hash == second.config_hash


def test_7b_a_different_seed_gives_a_different_parquet(learner_config, tmp_path):
    """Guards against the determinism test passing because nothing is random."""
    cfg = dataclasses.replace(
        learner_config,
        run=dataclasses.replace(learner_config.run, episodes=15, conditions=(TRADITIONAL,)),
    )
    units = units_at_difficulty(3, cfg, tmp_path, "seed_units")
    other = dataclasses.replace(
        cfg, seeds=dataclasses.replace(cfg.seeds, master=cfg.seeds.master + 1)
    )
    first = simulate(cfg, units, tmp_path, "seed_a")
    second = simulate(other, units, tmp_path, "seed_b")
    assert sha256_file(first.paths.learner_state_parquet) != sha256_file(
        second.paths.learner_state_parquet
    )


def test_7c_a_partial_condition_list_reproduces_the_full_run(learner_config, tmp_path):
    """Per-condition substreams mean a partial run is comparable to a full one."""
    units_cfg = dataclasses.replace(
        learner_config, run=dataclasses.replace(learner_config.run, episodes=10)
    )
    units = units_at_difficulty(3, units_cfg, tmp_path, "partial_units")

    full = simulate(
        dataclasses.replace(
            units_cfg, run=dataclasses.replace(units_cfg.run, conditions=CONDITIONS)
        ),
        units,
        tmp_path,
        "full",
    )
    alone = simulate(
        dataclasses.replace(
            units_cfg, run=dataclasses.replace(units_cfg.run, conditions=(TRADITIONAL,))
        ),
        units,
        tmp_path,
        "alone",
    )

    full_traditional = full.diagnostics[full.diagnostics["condition"] == TRADITIONAL]
    np.testing.assert_allclose(
        full_traditional["mean_effort"].to_numpy(),
        alone.diagnostics["mean_effort"].to_numpy(),
    )


# ---------------------------------------------------------------------------
# 8. Condition separation after support removal (§10.6)
# ---------------------------------------------------------------------------


def test_8_condition_separation_is_recorded_either_way(validation_run):
    """Scaffolding and substitution must be distinguishable once support is gone.

    **This test does not fail when they are not.** §10.6 makes an absent
    difference a falsification signal about the model rather than a broken test,
    so the verdict is written into `outputs/tables/phase3_validation_report.md`
    and surfaced as a warning. What *is* asserted is that the verdict was computed
    and recorded -- silence is the failure mode this guards against.
    """
    cfg, result = validation_run
    verdicts = condition_separation(result.checkpoints, cfg)
    assert verdicts, "separation was not assessed even though both arms are present"

    report = result.paths.validation_report.read_text(encoding="utf-8")
    assert "§10.6 condition separation after support removal" in report
    for verdict in verdicts:
        assert str(verdict.checkpoint_episode) in report

    unseparated = [verdict for verdict in verdicts if not verdict.separated]
    if unseparated:
        warnings.warn(
            f"§10.6 falsification signal: scaffolding and substitution are not "
            f"distinguishable at {len(unseparated)} of {len(verdicts)} checkpoint(s). "
            f"Recorded in {result.paths.validation_report}.",
            RuntimeWarning,
            stacklevel=2,
        )
        assert "falsification signal" in report


def test_8b_substitution_underperforms_after_support_removal(validation_run):
    """The direction the study predicts, at the final checkpoint.

    Reported as an assertion because a *reversal* -- substitution outperforming
    scaffolding unaided -- would mean the effort mechanism is wired backwards,
    which is a defect rather than a falsification.
    """
    _, result = validation_run
    last = result.checkpoints["checkpoint_episode"].max()
    window = result.checkpoints[result.checkpoints["checkpoint_episode"] == last]
    scaffolding = window[window["condition"] == AI_SCAFFOLDING]["unaided_accuracy_trained"].mean()
    substitution = window[window["condition"] == AI_SUBSTITUTION]["unaided_accuracy_trained"].mean()
    assert substitution <= scaffolding, (
        f"substitution ({substitution:.4f}) outperformed scaffolding ({scaffolding:.4f}) "
        f"with support removed -- check the sign of the eq. (19)/(23) terms"
    )


# ---------------------------------------------------------------------------
# Run hygiene
# ---------------------------------------------------------------------------


def test_the_run_writes_every_declared_output(validation_run):
    _, result = validation_run
    for path in (
        result.paths.responses_csv,
        result.paths.learner_state_parquet,
        result.paths.checkpoints_csv,
        result.paths.validation_report,
        result.paths.run_log,
    ):
        assert path.exists(), path


def test_the_run_log_records_provenance(validation_run):
    import json

    _, result = validation_run
    log = json.loads(result.paths.run_log.read_text(encoding="utf-8"))
    assert log["config_hash"] == result.config_hash
    assert log["master_seed"] == result.master_seed
    assert log["row_counts"]["responses"] > 0
    assert "clip_counts" in log
    assert "packages" in log and log["packages"]["numpy"]
    assert log["wall_time_s"] > 0


def test_a_placeholder_run_is_labelled_as_such(validation_run):
    """No output derived from the placeholder corpus may read as a result."""
    _, result = validation_run
    assert result.placeholder_corpus
    report = result.paths.validation_report.read_text(encoding="utf-8")
    assert "PLACEHOLDER CORPUS" in report
    assert "No number in this report is a project result" in report


def test_clipping_stays_modest_at_the_default_parameters(validation_run):
    """Heavy clipping means the parameters are wrong, not that learners are unusual.

    The threshold is loose on purpose: eq. (23) and eq. (25) have no saturating
    term, so some clipping is structural rather than a parameter error. The
    supervisor question is recorded in `docs/phase3_assumptions.md`.
    """
    _, result = validation_run
    heavy = {
        name: values["share"]
        for name, values in result.clip_counts.items()
        if values["share"] > 0.25
    }
    assert heavy == {}, f"heavily clipped state component(s): {heavy}"
