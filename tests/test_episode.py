"""The instructional episode (§3.1): order of operations and the ladder walk.

Two properties carry the design. First, §3.1's order must hold -- a tutor that
acted before seeing an attempt would be the `ai_substitution` condition by
accident and the study's contrast would collapse. Second, no loop may run over
learners: at 5,000 learners x 120 episodes x 2,000 draws a per-learner loop is
not slow, it is infeasible.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from src.learners.config import (
    AI_SCAFFOLDING,
    AI_SUBSTITUTION,
    STATE_LOWER,
    STATIC_AI,
    TRADITIONAL,
)
from src.learners.engine import STAGE_FIRST, STAGE_SUPPORTED, STAGE_TRANSFER, build_engine
from src.learners.episode import (
    RESPONSE_COLUMNS,
    STATE_COLUMNS,
    episode_id,
    run_episode,
)
from src.learners.support import TUTOR_STREAM
from src.learners.updates import ClipCounter
from tests.conftest import PROJECT_ROOT

ALL_CONDITIONS = (TRADITIONAL, AI_SCAFFOLDING, AI_SUBSTITUTION, STATIC_AI)


def run_one(state, unit, condition, learner_config, learner_streams, episode=0):
    engine = build_engine(learner_config, learner_streams)
    rng = learner_streams.generator(TUTOR_STREAM)
    counter = ClipCounter()
    return run_episode(
        state, unit, condition, engine, learner_config, rng, episode=episode, counter=counter
    )


# ---------------------------------------------------------------------------
# Shape and schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("condition", ALL_CONDITIONS)
def test_every_condition_runs(condition, learner_state, unit_views, learner_config, learner_streams):
    _, records = run_one(learner_state, unit_views[0], condition, learner_config, learner_streams)
    assert records.n_response_rows > 0
    assert set(records.responses) == set(RESPONSE_COLUMNS)
    assert set(records.state) == set(STATE_COLUMNS)


def test_state_table_has_one_row_per_learner(learner_state, unit_views, learner_config, learner_streams):
    _, records = run_one(learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams)
    assert records.state["learner_id"].size == learner_state.n_learners
    assert (records.state["time"] == 0).all()


def test_episode_id_is_shared_across_conditions(unit_views):
    """§3.1: the episode_id is constant across conditions for the same unit."""
    identifiers = {episode_id(7, unit_views[0]) for _ in ALL_CONDITIONS}
    assert len(identifiers) == 1
    assert unit_views[0].unit_id in identifiers.pop()


def test_episode_id_changes_with_the_episode_index(unit_views):
    assert episode_id(0, unit_views[0]) != episode_id(1, unit_views[0])


def test_state_advances_in_place_and_is_returned(learner_state, unit_views, learner_config, learner_streams):
    before = learner_state.K.copy()
    returned, _ = run_one(learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams)
    assert returned is learner_state
    assert learner_state.episodes_completed == 1
    assert not np.allclose(learner_state.K, before)


# ---------------------------------------------------------------------------
# Order of operations
# ---------------------------------------------------------------------------


def test_the_first_response_is_always_unaided(learner_state, unit_views, learner_config, learner_streams):
    for condition in ALL_CONDITIONS:
        _, records = run_one(
            learner_state.copy(), unit_views[0], condition, learner_config, learner_streams
        )
        first = records.responses["stage"] == STAGE_FIRST
        if not first.any():
            continue
        prompts = records.responses["prompt"][first]
        assert all(prompt.endswith("h=0.00") for prompt in prompts)


def test_the_transfer_question_is_always_unaided(learner_state, unit_views, learner_config, learner_streams):
    for condition in ALL_CONDITIONS:
        _, records = run_one(
            learner_state.copy(), unit_views[0], condition, learner_config, learner_streams
        )
        transfer = records.responses["stage"] == STAGE_TRANSFER
        assert transfer.sum() == learner_state.n_learners
        assert all(prompt.endswith("h=0.00") for prompt in records.responses["prompt"][transfer])


def test_substitution_requires_no_attempt(learner_state, unit_views, learner_config, learner_streams):
    """Some learners request help before producing any work, so have no unaided row."""
    _, records = run_one(
        learner_state, unit_views[0], AI_SUBSTITUTION, learner_config, learner_streams
    )
    first_rows = int((records.responses["stage"] == STAGE_FIRST).sum())
    assert 0 < first_rows < learner_state.n_learners
    assert records.diagnostics["attempt_rate"] < 1.0


def test_interactive_conditions_always_require_an_attempt(
    learner_state, unit_views, learner_config, learner_streams
):
    for condition in (TRADITIONAL, AI_SCAFFOLDING, STATIC_AI):
        _, records = run_one(
            learner_state.copy(), unit_views[0], condition, learner_config, learner_streams
        )
        assert records.diagnostics["attempt_rate"] == 1.0
        assert int((records.responses["stage"] == STAGE_FIRST).sum()) == learner_state.n_learners


def test_only_learners_who_needed_support_have_a_supported_row(
    learner_state, unit_views, learner_config, learner_streams
):
    """A learner who was right first time has no supported response to record."""
    _, records = run_one(
        learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams
    )
    supported = int((records.responses["stage"] == STAGE_SUPPORTED).sum())
    assert 0 < supported < learner_state.n_learners


# ---------------------------------------------------------------------------
# The mechanism the study rests on
# ---------------------------------------------------------------------------


def test_substitution_collapses_effort_relative_to_scaffolding(
    learner_state, unit_views, learner_config, learner_streams
):
    _, scaffolding = run_one(
        learner_state.copy(), unit_views[0], AI_SCAFFOLDING, learner_config, learner_streams
    )
    _, substitution = run_one(
        learner_state.copy(), unit_views[0], AI_SUBSTITUTION, learner_config, learner_streams
    )
    assert substitution.diagnostics["mean_effort"] < scaffolding.diagnostics["mean_effort"]
    assert substitution.diagnostics["mean_offloading"] > scaffolding.diagnostics["mean_offloading"]
    assert substitution.diagnostics["mean_answer_provided"] > scaffolding.diagnostics[
        "mean_answer_provided"
    ]


def test_scaffolding_is_more_effective_than_a_prewritten_ladder(
    learner_state, unit_views, learner_config, learner_streams
):
    """The contrast runs through adaptation -> eq. (20), not through support level."""
    _, traditional = run_one(
        learner_state.copy(), unit_views[0], TRADITIONAL, learner_config, learner_streams
    )
    _, scaffolding = run_one(
        learner_state.copy(), unit_views[0], AI_SCAFFOLDING, learner_config, learner_streams
    )
    assert (
        scaffolding.diagnostics["mean_effectiveness"]
        > traditional.diagnostics["mean_effectiveness"]
    )


# ---------------------------------------------------------------------------
# Record content
# ---------------------------------------------------------------------------


def test_confidence_and_correctness_are_in_range(
    learner_state, unit_views, learner_config, learner_streams
):
    _, records = run_one(learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams)
    confidence = records.responses["confidence"]
    assert confidence.min() >= STATE_LOWER
    assert confidence.max() <= 1.0
    assert set(np.unique(records.responses["correctness"])) <= {0, 1}


def test_latency_is_positive_and_rises_with_hints(
    learner_state, unit_views, learner_config, learner_streams
):
    _, records = run_one(learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams)
    latency = records.responses["latency_proxy"]
    assert latency.min() > 0
    supported = records.responses["stage"] == STAGE_SUPPORTED
    transfer = records.responses["stage"] == STAGE_TRANSFER
    assert latency[supported].mean() > latency[transfer].mean()


def test_answer_descriptors_use_the_unit_declared_values(
    learner_state, unit_views, learner_config, learner_streams
):
    """No answer is invented; a wrong answer is the unit's documented misconception."""
    unit = unit_views[0]
    _, records = run_one(learner_state, unit, TRADITIONAL, learner_config, learner_streams)
    answers = set(np.unique(records.responses["answer"]))
    assert answers <= {unit.reference_answer, unit.misconception_answer}
    assert set(np.unique(records.responses["response"])) <= {"correct", "misconception"}


def test_prompt_descriptor_names_the_unit_and_stage(
    learner_state, unit_views, learner_config, learner_streams
):
    _, records = run_one(learner_state, unit_views[0], TRADITIONAL, learner_config, learner_streams)
    prompt = records.responses["prompt"][0]
    assert unit_views[0].unit_id in prompt
    assert TRADITIONAL in prompt
    assert records.responses["token_count"].min() > 0


# ---------------------------------------------------------------------------
# Vectorisation
# ---------------------------------------------------------------------------


def test_no_module_loops_over_learners():
    """A `for` over an array of learners is a bug, not a slow path.

    Scans for a loop whose iterable is a learner-shaped expression. Loops over
    hint depths, conditions, episodes, bins and items are all fine and expected --
    they are bounded by small constants, not by the population size.
    """
    forbidden = {"learner_id", "n_learners", "learners", "state_matrix"}
    offenders = []
    for path in sorted((PROJECT_ROOT / "src" / "learners").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            names = {
                child.attr if isinstance(child, ast.Attribute) else child.id
                for child in ast.walk(node.iter)
                if isinstance(child, (ast.Name, ast.Attribute))
            }
            if names & forbidden:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"per-learner loop(s) found: {offenders}"


def test_runtime_scales_with_learners_not_worse(unit_views, learner_config, learner_streams):
    """A hidden per-learner loop would show up as super-linear growth."""
    import time
    from dataclasses import replace

    from src.learners.population import init_population, init_state

    timings = {}
    for n_learners in (500, 5000):
        cfg = replace(
            learner_config, population=replace(learner_config.population, n_learners=n_learners)
        )
        state = init_state(init_population(n_learners, cfg, learner_streams))
        engine = build_engine(cfg, learner_streams)
        rng = learner_streams.generator(TUTOR_STREAM)
        counter = ClipCounter()
        started = time.perf_counter()
        for episode in range(3):
            run_episode(
                state,
                unit_views[episode % len(unit_views)],
                TRADITIONAL,
                engine,
                cfg,
                rng,
                episode=episode,
                counter=counter,
            )
        timings[n_learners] = time.perf_counter() - started

    # 10x the learners must not cost far more than 10x the time. The bound is
    # loose because the fixed per-episode overhead dominates at 500 learners.
    assert timings[5000] < timings[500] * 30, timings
