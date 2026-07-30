"""Learner initialisation: the copula, the strata, and eq. (16).

Truncation, not clipping, is the property worth testing hardest. Clipping a
normal to [0, 1] piles probability mass onto the two bounds, which would surface
later as a population of learners with exactly zero dependence -- an artefact
that looks like a finding.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.learners.config import STATE_COMPONENTS, STATE_LOWER, STATE_UPPER
from src.learners.population import (
    POPULATION_COLUMNS,
    init_population,
    init_state,
    stratum_summary,
    theta_from_k,
)
from src.learners.seeds import SeedStreams


def test_returns_expected_shape_and_columns(population_frame, learner_config):
    assert len(population_frame) == learner_config.population.n_learners
    assert list(population_frame.columns) == list(POPULATION_COLUMNS)
    assert population_frame.index.name == "learner_id"


def test_every_state_component_is_inside_the_eq_15_bounds(population_frame):
    for name in STATE_COMPONENTS:
        values = population_frame[name].to_numpy()
        assert values.min() >= STATE_LOWER
        assert values.max() <= STATE_UPPER


def test_marginals_are_truncated_not_clipped(learner_config, learner_streams):
    """No mass should pile up exactly on the bounds."""
    frame = init_population(4000, learner_config, learner_streams)
    for name in STATE_COMPONENTS:
        values = frame[name].to_numpy()
        on_bound = np.count_nonzero((values == STATE_LOWER) | (values == STATE_UPPER))
        assert on_bound == 0, f"{name} has {on_bound} values sitting exactly on a bound"


def test_correlation_structure_is_reproduced(learner_config, learner_streams):
    """The copula must recover the sign of the configured correlations."""
    frame = init_population(6000, learner_config, learner_streams)
    matrix = frame.loc[:, list(STATE_COMPONENTS)].corr().to_numpy()
    target = np.asarray(learner_config.population.correlation)
    n = len(STATE_COMPONENTS)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(target[i, j]) < 0.15:
                continue
            assert np.sign(matrix[i, j]) == np.sign(target[i, j]), (
                f"correlation sign for {STATE_COMPONENTS[i]}/{STATE_COMPONENTS[j]} is wrong"
            )
            # Truncated marginals attenuate a Gaussian copula's correlation, so
            # the target is an upper bound rather than a value to match exactly.
            assert abs(matrix[i, j]) <= abs(target[i, j]) + 0.1


def test_stratum_weights_are_respected(learner_config, learner_streams):
    frame = init_population(8000, learner_config, learner_streams)
    shares = frame["stratum_index"].value_counts(normalize=True).sort_index().to_numpy()
    np.testing.assert_allclose(shares, learner_config.population.stratum_weights, atol=0.02)


def test_strata_shift_k_and_m_monotonically(learner_config, learner_streams):
    frame = init_population(8000, learner_config, learner_streams)
    summary = stratum_summary(frame)
    assert summary["K"].is_monotonic_increasing
    assert summary["M"].is_monotonic_increasing
    # Nothing else differs between strata.
    for name in ("R", "D"):
        spread = summary[name].max() - summary[name].min()
        assert spread < 0.05, f"{name} differs across strata but only K and M should"


def test_per_learner_parameters_have_the_declared_support(population_frame):
    """eq. (16): alpha_i > 0 (LogNormal), delta_i in [0, 1] (Beta)."""
    assert (population_frame["alpha_i"] > STATE_LOWER).all()
    assert (population_frame["delta_i"] >= STATE_LOWER).all()
    assert (population_frame["delta_i"] <= STATE_UPPER).all()
    assert (population_frame["speed_factor"] > STATE_LOWER).all()


def test_theta_0_matches_the_documented_map(population_frame, learner_config):
    expected = theta_from_k(population_frame["K"].to_numpy(), learner_config.population)
    np.testing.assert_allclose(population_frame["theta_0"].to_numpy(), expected)


def test_theta_is_increasing_in_knowledge(learner_config):
    values = theta_from_k(np.array([0.1, 0.5, 0.9]), learner_config.population)
    assert np.all(np.diff(values) > 0)


def test_same_seed_gives_an_identical_population(learner_config):
    first = init_population(
        200, learner_config, SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    )
    second = init_population(
        200, learner_config, SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    )
    assert first.equals(second)


def test_a_different_seed_gives_a_different_population(learner_config):
    first = init_population(
        200, learner_config, SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    )
    second = init_population(
        200,
        learner_config,
        SeedStreams(learner_config.seeds.master + 1, learner_config.seeds.streams),
    )
    assert not np.allclose(first["K"].to_numpy(), second["K"].to_numpy())


def test_zero_learners_raises(learner_config, learner_streams):
    with pytest.raises(ValueError, match="n_learners must be >= 1"):
        init_population(0, learner_config, learner_streams)


# ---------------------------------------------------------------------------
# LearnerState
# ---------------------------------------------------------------------------


def test_state_copy_is_independent(learner_state):
    clone = learner_state.copy()
    clone.K[:] = STATE_UPPER
    clone.independent_success_streak[:] = 99
    assert not np.allclose(learner_state.K, clone.K)
    assert learner_state.independent_success_streak.max() == 0


def test_state_starts_with_no_forecasts(learner_state):
    assert learner_state.brier_count.max() == 0
    assert learner_state.episodes_completed == 0


def test_state_matrix_is_in_canonical_order(learner_state):
    matrix = learner_state.state_matrix()
    assert matrix.shape == (learner_state.n_learners, len(STATE_COMPONENTS))
    for index, name in enumerate(STATE_COMPONENTS):
        np.testing.assert_allclose(matrix[:, index], getattr(learner_state, name))


def test_init_state_rejects_an_incomplete_frame(population_frame):
    with pytest.raises(ValueError, match="missing column"):
        init_state(population_frame.drop(columns=["alpha_i"]))


def test_init_state_does_not_alias_the_frame(population_frame):
    state = init_state(population_frame)
    state.K[:] = STATE_UPPER
    assert population_frame["K"].to_numpy().max() < STATE_UPPER


# ---------------------------------------------------------------------------
# Seed streams
# ---------------------------------------------------------------------------


def test_named_streams_are_independent(learner_config):
    streams = SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    population = streams.generator("population").random(100)
    responses = streams.generator("responses").random(100)
    assert not np.allclose(population, responses)


def test_generator_is_memoised_so_draws_advance(learner_streams):
    first = learner_streams.generator("responses").random(10)
    second = learner_streams.generator("responses").random(10)
    assert not np.allclose(first, second)


def test_derived_families_are_independent_of_sibling_labels(learner_config):
    """A condition's draws must not depend on which other conditions are run."""
    base = SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    alone = base.derive("traditional").generator("responses").random(50)
    other = SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)
    other.derive("ai_substitution").generator("responses").random(50)
    alongside = other.derive("traditional").generator("responses").random(50)
    np.testing.assert_allclose(alone, alongside)


def test_label_key_is_process_stable():
    """`hashlib`, not `hash()`: the built-in is salted per process."""
    from src.learners.seeds import label_key

    assert label_key("traditional") == label_key("traditional")
    assert label_key("traditional") != label_key("ai_scaffolding")
    assert label_key("traditional") >= 0


def test_unknown_stream_raises(learner_streams):
    with pytest.raises(KeyError, match="unknown seed stream"):
        learner_streams.generator("plasticity")
