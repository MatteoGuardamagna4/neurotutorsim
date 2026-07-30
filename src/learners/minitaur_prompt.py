"""Psych-101 prompt construction for the Minitaur feasibility probe (§6.1, §10.2).

**Minitaur is a next-choice predictor, not a dialogue partner.** It was trained
on Psych-101, a corpus of psychology experiments in which the human participant's
choice on each trial is wrapped in `<<` `>>` tokens, and it is used with
`max_new_tokens=1`. Asked to solve a break-even problem it will not produce a
calculation; asked to continue a transcript that ends in `<<` it will produce one
token, which is the choice.

So the probe **scores** a small discrete choice set rather than generating text:

======  ==========================================================
`A`     the correct answer
`B`     the unit's documented misconception
`C`     request a hint
======  ==========================================================

Summing the log-likelihood of each option's tokens under teacher forcing and
softmaxing over the set gives `P(choice)`, and `P(A)` is directly comparable to
the transparent baseline's `p_correct`.

Observable history only
-----------------------
A prompt encodes what a participant could have observed: prior answers, whether
they were right, whether a hint was used, and reported confidence. It must
**never** contain the latent state -- `K`, `M`, `R`, `C`, `D`, `theta`, `alpha`,
`delta`, `b_u`. Leaking those would hand the model the answer to the question the
probe is asking, and the resulting agreement with the baseline would be an
artefact of the prompt. `find_latent_leaks` is the rule in executable form, and
`tests/test_minitaur_prompt.py` asserts every generated prompt is clean.

**Standard library only.** No numpy, no pandas, no config import, and certainly
no torch: `scripts/benchmark_minitaur.py` runs on Colab and this is the only
module of `src/learners/` it is allowed to import, precisely because importing it
cannot drag Track B's stack onto a GPU box.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

#: Choice keys. `A`/`B`/`C` are the §6.1 set; the keys are deliberately short
#: because the model emits exactly one token after the final `<<`.
CHOICE_CORRECT = "A"
CHOICE_MISCONCEPTION = "B"
CHOICE_HINT = "C"

#: Open-choice marker the prompt ends with, and its closing counterpart.
OPEN_MARKER = "<<"
CLOSE_MARKER = ">>"

#: Latent quantities that may never appear in a prompt.
#:
#: Multi-character names are matched on word boundaries. The single letters are
#: matched only in an *assignment* context (`K=0.4`, `C: 0.9`) because `A`, `B`
#: and `C` are choice keys and `D`/`M`/`R` occur inside ordinary words -- a
#: blanket letter search would either fire on every prompt or on none.
_LATENT_WORD_PATTERN = re.compile(
    r"\b(theta|alpha_i|alpha|delta_i|delta|b_u|p_correct|brier|"
    r"knowledge state|memory strength|independent reasoning|calibration|dependence|"
    r"latent|ability)\b",
    re.IGNORECASE,
)
_LATENT_ASSIGNMENT_PATTERN = re.compile(r"\b([KMRCD])\s*[=:]\s*-?\d")


@dataclass(frozen=True)
class Choice:
    """One option in the choice set."""

    key: str
    label: str

    def __post_init__(self) -> None:
        if not self.key or len(self.key) != 1:
            raise ValueError(f"a choice key must be a single character; got {self.key!r}")
        if not self.label.strip():
            raise ValueError(f"choice {self.key!r} has an empty label")


@dataclass(frozen=True)
class ChoiceSet:
    """The discrete options offered on a trial."""

    choices: Tuple[Choice, ...]

    def __post_init__(self) -> None:
        if len(self.choices) < 2:
            raise ValueError(
                f"a choice set needs at least two options for a softmax over it to mean "
                f"anything; got {len(self.choices)}"
            )
        keys = [choice.key for choice in self.choices]
        if len(set(keys)) != len(keys):
            raise ValueError(f"choice keys must be unique; got {keys}")

    @property
    def keys(self) -> Tuple[str, ...]:
        return tuple(choice.key for choice in self.choices)

    def rendered(self) -> str:
        return "  ".join(f"{choice.key}) {choice.label}" for choice in self.choices)


@dataclass(frozen=True)
class EpisodeTurn:
    """One observable event in the transcript.

    Every field is something a participant could have seen. There is no field for
    latent state, so a leak has to be introduced through `problem_text` -- which
    is what `find_latent_leaks` checks.
    """

    problem_text: str
    chosen: Optional[str] = None
    correct: Optional[bool] = None
    hint_given: bool = False
    answer_revealed: bool = False
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        if not self.problem_text.strip():
            raise ValueError("problem_text must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must lie in [0, 1]; got {self.confidence}")


@dataclass(frozen=True)
class EpisodeHistory:
    """The trials already completed, oldest first."""

    turns: Tuple[EpisodeTurn, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for index, turn in enumerate(self.turns):
            if turn.chosen is None:
                raise ValueError(
                    f"history turn {index} has no recorded choice. History is what the "
                    f"participant already did; an open trial belongs in `current` instead."
                )


def choice_set_for_unit(
    *,
    reference_answer: Optional[str] = None,
    misconception_answer: Optional[str] = None,
    misconception_description: Optional[str] = None,
    hint_label: str = "Ask for a hint",
) -> ChoiceSet:
    """Build the §6.1 three-option choice set from a unit's declared fields.

    A unit with no documented misconception still gets a `B` option, labelled as
    an unspecified alternative answer. Dropping `B` instead would change the size
    of the softmax between units and make `P(A)` incomparable across the corpus.
    """
    correct_label = (
        f"The answer is {reference_answer}" if reference_answer else "The correct answer"
    )
    if misconception_answer and misconception_description:
        wrong_label = f"The answer is {misconception_answer} ({misconception_description})"
    elif misconception_answer:
        wrong_label = f"The answer is {misconception_answer}"
    else:
        wrong_label = "A different answer"
    return ChoiceSet(
        (
            Choice(CHOICE_CORRECT, correct_label),
            Choice(CHOICE_MISCONCEPTION, wrong_label),
            Choice(CHOICE_HINT, hint_label),
        )
    )


def _render_turn(index: int, turn: EpisodeTurn) -> str:
    lines = [f"Problem {index}: {turn.problem_text.strip()}"]
    if turn.hint_given:
        lines.append("You were given a hint before answering.")
    lines.append(f"You chose {OPEN_MARKER}{turn.chosen}{CLOSE_MARKER}.")
    if turn.confidence is not None:
        lines.append(f"You reported {round(turn.confidence * 100)}% confidence in that choice.")
    if turn.correct is not None:
        lines.append("That answer was correct." if turn.correct else "That answer was incorrect.")
    if turn.answer_revealed:
        lines.append("The correct answer was then shown to you.")
    return "\n".join(lines)


def to_psych101_prompt(
    episode_history: EpisodeHistory,
    choice_set: ChoiceSet,
    *,
    current: EpisodeTurn,
    instructions: Optional[str] = None,
) -> str:
    """Transcribe an episode as a Psych-101 trial sequence ending in an open choice.

    Parameters
    ----------
    episode_history
        Completed trials, oldest first. Every turn must carry the choice made.
    choice_set
        Options offered on the open trial.
    current
        The open trial. Its `chosen` field is ignored -- that is the thing being
        predicted.
    instructions
        Overrides the default task framing.

    Returns
    -------
    str
        A prompt ending with the bare `<<` marker, so that the model's single next
        token is the choice key. Nothing follows the marker: a trailing space or
        newline changes the tokenisation and therefore the scores.

    Raises
    ------
    ValueError
        If the assembled prompt contains a latent variable (see
        `find_latent_leaks`). The check runs on the finished string, so a leak
        introduced through any field is caught rather than only one introduced
        through a known one.
    """
    header = instructions or (
        "You are taking part in a study on learning business problem-solving. "
        "On each trial you see a problem and choose one option. "
        "Your choices are recorded between double angle brackets."
    )
    blocks = [header, ""]
    for index, turn in enumerate(episode_history.turns, start=1):
        blocks.append(_render_turn(index, turn))
        blocks.append("")
    open_index = len(episode_history.turns) + 1
    blocks.append(f"Problem {open_index}: {current.problem_text.strip()}")
    if current.hint_given:
        blocks.append("You were given a hint before answering.")
    blocks.append(f"Options: {choice_set.rendered()}")
    blocks.append(f"You choose {OPEN_MARKER}")
    prompt = "\n".join(blocks)

    leaks = find_latent_leaks(prompt)
    if leaks:
        raise ValueError(
            f"the generated prompt leaks latent state: {list(leaks)}. A Minitaur prompt encodes "
            f"observable history only -- prior answers, errors, hint usage, confidence. Handing "
            f"the model the learner's latent state would make any agreement with the transparent "
            f"baseline an artefact of the prompt rather than a validation of the baseline."
        )
    return prompt


def find_latent_leaks(text: str) -> Tuple[str, ...]:
    """Latent variable names found in `text`, deduplicated and sorted.

    Two patterns, for the reason given in the module docstring: multi-character
    names on word boundaries, single-letter state names only where they are being
    assigned a number.
    """
    found = {match.group(0) for match in _LATENT_WORD_PATTERN.finditer(text)}
    found |= {match.group(0) for match in _LATENT_ASSIGNMENT_PATTERN.finditer(text)}
    return tuple(sorted(found))


def scored_continuations(choice_set: ChoiceSet) -> Tuple[str, ...]:
    """The strings whose log-likelihood the probe scores, one per option.

    Each is the choice key followed by the closing marker, which is what a
    Psych-101 transcript actually contains after `<<`. Scoring the key alone would
    leave the model free to spend probability on continuing a longer token.
    """
    return tuple(f"{choice.key}{CLOSE_MARKER}" for choice in choice_set.choices)


def summarise_prompt(prompt: str) -> dict:
    """Cheap descriptive stats, for the feasibility report's prompt-length column."""
    return {
        "characters": len(prompt),
        "whitespace_tokens": len(prompt.split()),
        "trials": prompt.count("Problem "),
        "recorded_choices": prompt.count(CLOSE_MARKER),
        "ends_with_open_marker": prompt.endswith(OPEN_MARKER),
    }


def history_from_records(records: Sequence[dict]) -> EpisodeHistory:
    """Build a history from plain dict records.

    Accepts the observable subset only. An unexpected key raises rather than being
    ignored, so a caller cannot pass a full state row and have the latent fields
    silently dropped -- they would be dropped this time and leak the next time the
    renderer grows a field.
    """
    allowed = {
        "problem_text",
        "chosen",
        "correct",
        "hint_given",
        "answer_revealed",
        "confidence",
    }
    turns = []
    for index, record in enumerate(records):
        unknown = sorted(set(record) - allowed)
        if unknown:
            raise ValueError(
                f"record {index} carries key(s) {unknown}, which are not observable trial "
                f"fields. Allowed: {sorted(allowed)}."
            )
        turns.append(EpisodeTurn(**record))
    return EpisodeHistory(tuple(turns))
