"""Psych-101 prompt construction (§6.1). Pure CPU, standard library only.

The test that matters most is the latent-leak one. If a prompt carried `K`,
`theta` or `alpha`, Minitaur would be handed the answer to the question the §10.2
probe is asking, and any agreement with the transparent baseline would be an
artefact of the prompt rather than a validation of the baseline.
"""

from __future__ import annotations

import pytest

from src.learners.minitaur_prompt import (
    CHOICE_CORRECT,
    CHOICE_HINT,
    CHOICE_MISCONCEPTION,
    CLOSE_MARKER,
    OPEN_MARKER,
    Choice,
    ChoiceSet,
    EpisodeHistory,
    EpisodeTurn,
    choice_set_for_unit,
    find_latent_leaks,
    history_from_records,
    scored_continuations,
    summarise_prompt,
    to_psych101_prompt,
)


@pytest.fixture
def choice_set() -> ChoiceSet:
    return choice_set_for_unit(
        reference_answer="42",
        misconception_answer="30",
        misconception_description="dividing fixed costs by price",
    )


@pytest.fixture
def history() -> EpisodeHistory:
    return EpisodeHistory(
        (
            EpisodeTurn(
                problem_text="How many units must it sell to break even?",
                chosen=CHOICE_CORRECT,
                correct=True,
                confidence=0.8,
            ),
            EpisodeTurn(
                problem_text="What price breaks even at 3,000 subscribers?",
                chosen=CHOICE_MISCONCEPTION,
                correct=False,
                hint_given=True,
                answer_revealed=True,
                confidence=0.4,
            ),
        )
    )


@pytest.fixture
def current() -> EpisodeTurn:
    return EpisodeTurn(problem_text="What contribution margin breaks even at 500 units?")


# ---------------------------------------------------------------------------
# The Psych-101 format
# ---------------------------------------------------------------------------


def test_prompt_ends_with_a_bare_open_marker(history, choice_set, current):
    """The model's single next token must be the choice key.

    A trailing space or newline changes the tokenisation and therefore the
    scores, so the assertion is on the exact final characters.
    """
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert prompt.endswith(OPEN_MARKER)
    assert not prompt.endswith(OPEN_MARKER + " ")
    assert not prompt.endswith("\n")


def test_prior_choices_are_wrapped_in_double_angle_brackets(history, choice_set, current):
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert f"{OPEN_MARKER}{CHOICE_CORRECT}{CLOSE_MARKER}" in prompt
    assert f"{OPEN_MARKER}{CHOICE_MISCONCEPTION}{CLOSE_MARKER}" in prompt
    # One closing marker per completed trial; the open trial has none.
    assert prompt.count(CLOSE_MARKER) == len(history.turns)


def test_the_open_trial_offers_the_choice_set(history, choice_set, current):
    prompt = to_psych101_prompt(history, choice_set, current=current)
    for choice in choice_set.choices:
        assert f"{choice.key}) {choice.label}" in prompt


def test_observable_history_is_transcribed(history, choice_set, current):
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert "That answer was correct." in prompt
    assert "That answer was incorrect." in prompt
    assert "You were given a hint before answering." in prompt
    assert "The correct answer was then shown to you." in prompt
    assert "80% confidence" in prompt


def test_trials_are_numbered_in_order(history, choice_set, current):
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert prompt.index("Problem 1:") < prompt.index("Problem 2:") < prompt.index("Problem 3:")


def test_an_empty_history_still_produces_an_open_trial(choice_set, current):
    prompt = to_psych101_prompt(EpisodeHistory(), choice_set, current=current)
    assert "Problem 1:" in prompt
    assert prompt.endswith(OPEN_MARKER)
    assert CLOSE_MARKER not in prompt


def test_summarise_prompt_reports_the_shape(history, choice_set, current):
    summary = summarise_prompt(to_psych101_prompt(history, choice_set, current=current))
    assert summary["trials"] == 3
    assert summary["recorded_choices"] == 2
    assert summary["ends_with_open_marker"] is True
    assert summary["whitespace_tokens"] > 0


# ---------------------------------------------------------------------------
# Latent state must never appear
# ---------------------------------------------------------------------------


def test_no_latent_variable_appears_in_a_generated_prompt(history, choice_set, current):
    prompt = to_psych101_prompt(history, choice_set, current=current)
    assert find_latent_leaks(prompt) == ()


@pytest.mark.parametrize(
    "leak",
    [
        "Your theta is 0.4.",
        "K = 0.62 at this point.",
        "C: 0.9 calibration so far.",
        "alpha_i governs how fast you learn.",
        "Your latent ability is high.",
        "Your knowledge state is 0.5.",
        "b_u for this item is 1.2.",
        "Your dependence has been rising.",
    ],
)
def test_a_leaked_latent_variable_is_detected(leak):
    assert find_latent_leaks(leak) != ()


def test_a_prompt_carrying_latent_state_raises(choice_set):
    leaking = EpisodeTurn(problem_text="Given your theta of 0.4, solve this.")
    with pytest.raises(ValueError, match="leaks latent state"):
        to_psych101_prompt(EpisodeHistory(), choice_set, current=leaking)


def test_the_leak_check_runs_on_the_finished_string(choice_set, current):
    """A leak introduced through any field must be caught, not only a known one."""
    with pytest.raises(ValueError, match="leaks latent state"):
        to_psych101_prompt(
            EpisodeHistory(),
            choice_set,
            current=current,
            instructions="You are a simulated learner whose knowledge state is tracked.",
        )


def test_the_choice_keys_are_not_mistaken_for_latent_names(choice_set, current):
    """`C` is both a choice key and the calibration state, so the check is contextual."""
    prompt = to_psych101_prompt(EpisodeHistory(), choice_set, current=current)
    assert f"{CHOICE_HINT})" in prompt
    assert find_latent_leaks(prompt) == ()


def test_history_from_records_rejects_non_observable_fields():
    with pytest.raises(ValueError, match="not observable trial fields"):
        history_from_records([{"problem_text": "x", "chosen": "A", "K": 0.5}])


def test_history_from_records_accepts_the_observable_subset():
    built = history_from_records(
        [{"problem_text": "x", "chosen": "A", "correct": True, "confidence": 0.6}]
    )
    assert len(built.turns) == 1


# ---------------------------------------------------------------------------
# Choice sets
# ---------------------------------------------------------------------------


def test_choice_set_has_the_three_documented_options(choice_set):
    assert choice_set.keys == (CHOICE_CORRECT, CHOICE_MISCONCEPTION, CHOICE_HINT)


def test_a_unit_without_a_misconception_still_gets_three_options():
    """Dropping B would change the softmax size and make P(A) incomparable."""
    built = choice_set_for_unit(reference_answer="42")
    assert len(built.choices) == 3
    assert "A different answer" in built.rendered()


def test_scored_continuations_include_the_closing_marker(choice_set):
    """Scoring the bare key would leave probability free to continue a longer token."""
    assert scored_continuations(choice_set) == (
        f"{CHOICE_CORRECT}{CLOSE_MARKER}",
        f"{CHOICE_MISCONCEPTION}{CLOSE_MARKER}",
        f"{CHOICE_HINT}{CLOSE_MARKER}",
    )


def test_a_one_option_choice_set_raises():
    with pytest.raises(ValueError, match="at least two options"):
        ChoiceSet((Choice("A", "only"),))


def test_duplicate_choice_keys_raise():
    with pytest.raises(ValueError, match="unique"):
        ChoiceSet((Choice("A", "one"), Choice("A", "two")))


def test_a_multi_character_choice_key_raises():
    with pytest.raises(ValueError, match="single character"):
        Choice("AB", "two characters")


def test_history_turns_must_record_a_choice():
    with pytest.raises(ValueError, match="no recorded choice"):
        EpisodeHistory((EpisodeTurn(problem_text="x"),))


def test_out_of_range_confidence_raises():
    with pytest.raises(ValueError, match="confidence must lie in"):
        EpisodeTurn(problem_text="x", chosen="A", confidence=1.5)


def test_empty_problem_text_raises():
    with pytest.raises(ValueError, match="must not be empty"):
        EpisodeTurn(problem_text="   ")


# ---------------------------------------------------------------------------
# Isolation
# ---------------------------------------------------------------------------


def test_the_module_imports_only_the_standard_library():
    """The Colab probe imports this and nothing else from src/learners/."""
    import ast

    from tests.conftest import PROJECT_ROOT

    tree = ast.parse(
        (PROJECT_ROOT / "src" / "learners" / "minitaur_prompt.py").read_text(encoding="utf-8")
    )
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "re", "dataclasses", "typing"}, imported
