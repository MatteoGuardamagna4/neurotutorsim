"""The units adapter, the `b_u` map, and the decision-gate-16 guard.

The gate-16 test is the important one in this file. `data/processed/units.csv`
holds a single unit and gate 16 requires ten, so a Phase III run over the real
corpus must fail. A clean run there would mean the guard went missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.learners.curriculum import (
    PROVENANCE_UNSPECIFIED,
    SYNTHETIC_PLACEHOLDER,
    difficulty_to_b_u,
    is_placeholder_corpus,
    load_curriculum,
    units_to_views,
)
from tests.conftest import PROJECT_ROOT, make_units_frame

REAL_UNITS_CSV = PROJECT_ROOT / "data" / "processed" / "units.csv"


# ---------------------------------------------------------------------------
# Decision gate 16
# ---------------------------------------------------------------------------


def test_gate_16_fires_on_a_one_row_real_units_table(tmp_path, learner_config):
    """A real corpus below ten units raises, naming decision gate 16."""
    frame = make_units_frame(1, provenance="authored")
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)

    with pytest.raises(ValueError) as error:
        load_curriculum(path, learner_config.curriculum)

    message = str(error.value)
    assert "decision gate 16" in message
    assert "1 unit(s)" in message
    assert f"{learner_config.curriculum.min_units} are required" in message
    assert "matching and correctness tests" in message
    assert "no override flag" in message.lower()


@pytest.mark.skipif(not REAL_UNITS_CSV.exists(), reason="the real units table is not present")
def test_gate_16_fires_on_the_actual_project_corpus(learner_config):
    """The committed corpus is below gate 16 today, so this must raise."""
    n_units = len(pd.read_csv(REAL_UNITS_CSV))
    if n_units >= learner_config.curriculum.min_units:
        pytest.skip(f"the corpus has grown to {n_units} units; gate 16 is cleared")
    with pytest.raises(ValueError, match="decision gate 16"):
        load_curriculum(REAL_UNITS_CSV, learner_config.curriculum)


def test_gate_16_waived_only_for_a_fully_placeholder_table(tmp_path, learner_config):
    path = tmp_path / "units.csv"
    make_units_frame(3, provenance=SYNTHETIC_PLACEHOLDER).to_csv(path, index=False)
    units = load_curriculum(path, learner_config.curriculum)
    assert len(units) == 3
    assert is_placeholder_corpus(units)


def test_a_mixed_table_below_the_threshold_still_raises(tmp_path, learner_config):
    """Padding a real corpus with placeholders must not clear the gate."""
    frame = make_units_frame(3, provenance=SYNTHETIC_PLACEHOLDER)
    frame.loc[0, "provenance"] = "authored"
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="decision gate 16"):
        load_curriculum(path, learner_config.curriculum)


def test_a_table_with_no_provenance_column_is_treated_as_real(tmp_path, learner_config):
    frame = make_units_frame(2).drop(columns=["provenance"])
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="decision gate 16"):
        load_curriculum(path, learner_config.curriculum)


def test_provenance_defaults_to_unspecified_above_the_threshold(tmp_path, learner_config):
    frame = make_units_frame(12).drop(columns=["provenance"])
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    units = load_curriculum(path, learner_config.curriculum)
    assert (units["provenance"] == PROVENANCE_UNSPECIFIED).all()
    assert not is_placeholder_corpus(units)


def test_load_curriculum_has_no_override_parameter():
    """The guard must not be bypassable by a keyword argument."""
    import inspect

    parameters = set(inspect.signature(load_curriculum).parameters)
    assert parameters == {"path", "cfg"}


# ---------------------------------------------------------------------------
# The b_u map
# ---------------------------------------------------------------------------


def test_b_u_is_monotone_increasing_in_difficulty(learner_config):
    difficulties = np.array([1, 2, 3, 4, 5])
    b_u = difficulty_to_b_u(difficulties, learner_config.curriculum)
    assert np.all(np.diff(b_u) > 0)


def test_b_u_is_zero_at_the_configured_centre(learner_config):
    cfg = learner_config.curriculum
    centred = difficulty_to_b_u(np.array([cfg.difficulty_center]), cfg)
    assert centred[0] == pytest.approx(cfg.b_intercept)


def test_b_u_step_matches_the_slope(learner_config):
    cfg = learner_config.curriculum
    b_u = difficulty_to_b_u(np.array([2, 3]), cfg)
    assert (b_u[1] - b_u[0]) == pytest.approx(cfg.b_slope)


def test_b_u_is_attached_by_the_loader(units_frame, learner_config):
    expected = difficulty_to_b_u(units_frame["difficulty"].to_numpy(), learner_config.curriculum)
    np.testing.assert_allclose(units_frame["b_u"].to_numpy(), expected)


# ---------------------------------------------------------------------------
# Table validation
# ---------------------------------------------------------------------------


def test_missing_required_column_raises(tmp_path, learner_config):
    frame = make_units_frame(12).drop(columns=["concept"])
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required column"):
        load_curriculum(path, learner_config.curriculum)


def test_duplicate_unit_id_raises(tmp_path, learner_config):
    frame = make_units_frame(12)
    frame.loc[1, "unit_id"] = frame.loc[0, "unit_id"]
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unit_id must be unique"):
        load_curriculum(path, learner_config.curriculum)


def test_out_of_range_difficulty_raises(tmp_path, learner_config):
    frame = make_units_frame(12)
    frame.loc[0, "difficulty"] = 9
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="difficulty outside the declared range"):
        load_curriculum(path, learner_config.curriculum)


def test_non_integer_difficulty_raises_rather_than_rounding(tmp_path, learner_config):
    frame = make_units_frame(12)
    frame["difficulty"] = frame["difficulty"].astype(float)
    frame.loc[0, "difficulty"] = 2.5
    path = tmp_path / "units.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non-integer difficulty"):
        load_curriculum(path, learner_config.curriculum)


def test_missing_file_raises(tmp_path, learner_config):
    with pytest.raises(FileNotFoundError, match="no default corpus"):
        load_curriculum(tmp_path / "absent.csv", learner_config.curriculum)


def test_unsupported_format_raises(tmp_path, learner_config):
    path = tmp_path / "units.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported units table format"):
        load_curriculum(path, learner_config.curriculum)


# ---------------------------------------------------------------------------
# UnitView
# ---------------------------------------------------------------------------


def test_units_to_views_requires_resolved_effectiveness_columns(units_frame):
    """Coverage must be defaulted by the function that logs the substitution."""
    with pytest.raises(ValueError, match="resolve_effectiveness_columns"):
        units_to_views(units_frame)


def test_unit_views_carry_the_optional_answer_columns(unit_views):
    view = unit_views[0]
    assert view.reference_answer is not None
    assert view.misconception_answer is not None
    assert 0.0 <= view.coverage <= 1.0
    assert view.provenance == SYNTHETIC_PLACEHOLDER


def test_no_unit_id_is_hard_coded_in_the_package():
    """§2: the engine must never name a unit outside of tests."""
    import re

    package = PROJECT_ROOT / "src" / "learners"
    pattern = re.compile(r"\bbe_\d{3}\b")
    offenders = [
        path.name
        for path in sorted(package.glob("*.py"))
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
