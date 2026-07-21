"""Content validation checks for tutor-transcript stimuli (§5.4).

Turn parsing
------------
A stimulus body is a sequence of `TUTOR:`/`LEARNER:` turns, one marker per
line, each followed by its (possibly multi-line) text up to the next marker.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from src.generation import features as F

_TURN_RE = re.compile(r"(?m)^(TUTOR|LEARNER):[ \t]*(.*(?:\n(?!(?:TUTOR|LEARNER):).*)*)")
_DIGIT_RE = re.compile(r"\d")


def _split_turns(body: str) -> List[Tuple[str, str]]:
    turns = [(m.group(1), m.group(2).strip()) for m in _TURN_RE.finditer(body)]
    if not turns:
        raise ValueError("no TUTOR:/LEARNER: turns found in body")
    return turns


def _normalize_numbers(text: str) -> str:
    """Strip thousands separators (commas, spaces) between digits, e.g.
    '4,000' and '4 000' both normalize to '4000'."""
    return re.sub(r"(?<=\d)[,\s](?=\d)", "", text)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LeakageResult:
    leaked: bool
    first_attempt_turn_index: int
    detail: str


def check_leakage(body: str, final_answer: Any) -> LeakageResult:
    """Does the final answer appear in a TUTOR turn before the learner's
    first attempt?

    The first attempt is the first `LEARNER:` turn containing a digit.
    Numbers are normalized (thousands separators stripped) before
    comparison, since '4,000' and '4000' must be treated as the same value.
    Scaffolding must not leak; substitution is expected to reveal early by
    design (§5.2.8), so a `True` result there is not a failure.
    """
    turns = _split_turns(body)
    final_answer_norm = _normalize_numbers(str(final_answer))

    first_attempt_idx = next(
        (i for i, (speaker, text) in enumerate(turns) if speaker == "LEARNER" and _DIGIT_RE.search(text)),
        None,
    )
    if first_attempt_idx is None:
        raise ValueError("no LEARNER turn containing a numeric attempt was found")

    preceding_tutor_text = " ".join(
        text for speaker, text in turns[:first_attempt_idx] if speaker == "TUTOR"
    )
    leaked = final_answer_norm in _normalize_numbers(preceding_tutor_text)

    return LeakageResult(
        leaked=leaked,
        first_attempt_turn_index=first_attempt_idx,
        detail=f"final_answer={final_answer_norm!r} found_before_attempt={leaked}",
    )


# --------------------------------------------------------------------------- #
# Tutor-turn classification
# --------------------------------------------------------------------------- #

def _classify_single_turn(text: str) -> str:
    """Rule, applied in order:

    1. `hint` -- the turn opens with the literal marker 'Hint:' (case
       insensitive). Hints are prewritten nudges, not open questions.
    2. `guiding_question` -- the turn contains a '?'. A question mark means
       the tutor is asking the learner to do the next step, not doing it
       for them.
    3. `answer_reveal` -- anything else: a declarative turn that neither
       hints nor asks, i.e. it is telling the learner something conclusive.
    """
    stripped = text.strip()
    if stripped.lower().startswith("hint:"):
        return "hint"
    if "?" in text:
        return "guiding_question"
    return "answer_reveal"


def classify_tutor_turns(body: str) -> List[str]:
    """Label each TUTOR: turn as guiding_question | hint | answer_reveal, in
    turn order. See `_classify_single_turn` for the rule."""
    turns = _split_turns(body)
    return [_classify_single_turn(text) for speaker, text in turns if speaker == "TUTOR"]


# --------------------------------------------------------------------------- #
# Readability
# --------------------------------------------------------------------------- #

def check_readability(bodies: Dict[str, str], threshold: float = 1.0) -> Dict[str, Any]:
    """FK grade per condition body, flagged if the spread across conditions
    exceeds `threshold` grade levels (default 1.0, §5.4)."""
    grades = {name: F.fk_grade(body) for name, body in bodies.items()}
    spread = max(grades.values()) - min(grades.values())
    return {"grades": grades, "spread": spread, "flagged": spread > threshold}
