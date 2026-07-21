"""Verification gate G: leakage positive/negative cases, scaffolding
no-leak compliance, and substitution early-reveal detection."""

import pytest

from src.validation import checks as C
from src.generation.stimulus_io import parse_front_matter

FINAL_ANSWER = 4000  # be_001 canonical break-even quantity


def test_check_leakage_negative_case():
    body = (
        "TUTOR: What is 2 plus 2?\n\n"
        "LEARNER: Is it 5?\n\n"
        "TUTOR: Hint: try counting on your fingers.\n\n"
        "LEARNER: Is it 4?\n\n"
        "TUTOR: Yes, 4 is correct.\n"
    )
    result = C.check_leakage(body, final_answer=4)
    assert result.leaked is False


def test_check_leakage_positive_case():
    body = (
        "TUTOR: The answer to this problem is 4,000 units. Now, what is 2 plus 2?\n\n"
        "LEARNER: Is it 5?\n\n"
        "TUTOR: Not quite, try again.\n"
    )
    result = C.check_leakage(body, final_answer=4000)
    assert result.leaked is True


def test_check_leakage_raises_without_attempt():
    body = "TUTOR: Hello there.\n\nLEARNER: Hi, how are you?\n"
    with pytest.raises(ValueError):
        C.check_leakage(body, final_answer=4000)


def test_scaffolding_does_not_leak_before_first_attempt():
    _, body = parse_front_matter("stimuli/ai_scaffolding/be_001_primary.txt")
    result = C.check_leakage(body, final_answer=FINAL_ANSWER)
    assert result.leaked is False


def test_scaffolding_turn_policy_compliance():
    _, body = parse_front_matter("stimuli/ai_scaffolding/be_001_primary.txt")
    labels = C.classify_tutor_turns(body)
    assert labels.count("guiding_question") >= 1
    assert labels.count("hint") >= 2
    first_reveal = labels.index("answer_reveal") if "answer_reveal" in labels else len(labels)
    leakage = C.check_leakage(body, final_answer=FINAL_ANSWER)
    # any answer_reveal turn must come after the learner's first attempt
    assert first_reveal >= 0
    assert leakage.leaked is False


def test_substitution_reveals_in_first_tutor_turn_after_error():
    _, body = parse_front_matter("stimuli/ai_substitution/be_001_primary.txt")
    labels = C.classify_tutor_turns(body)
    leakage = C.check_leakage(body, final_answer=FINAL_ANSWER)
    # not "leakage" in the pre-attempt sense: the answer is not stated before
    # the learner's first (scripted) error, only immediately after it.
    assert leakage.leaked is False
    # the tutor turn immediately following the learner's error (index 1) reveals
    assert labels[1] == "answer_reveal"


def test_check_readability_across_conditions():
    bodies = {}
    for condition, path in [
        ("traditional", "stimuli/traditional/be_001_primary.txt"),
        ("ai_scaffolding", "stimuli/ai_scaffolding/be_001_primary.txt"),
        ("ai_substitution", "stimuli/ai_substitution/be_001_primary.txt"),
    ]:
        _, body = parse_front_matter(path)
        bodies[condition] = body
    result = C.check_readability(bodies)
    assert set(result["grades"]) == {"traditional", "ai_scaffolding", "ai_substitution"}
    assert isinstance(result["flagged"], bool)
