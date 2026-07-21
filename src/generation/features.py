"""Observable-property feature vector for stimulus matching (§5.3).

Every feature here is computed deterministically from raw text so matching
diagnostics are reproducible without any external NLP dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?\n]+")
_ALPHA_TOKEN_RE = re.compile(r"[a-z]+")
_VOWELS = "aeiouy"


def words_list(body: str) -> List[str]:
    return _WORD_RE.findall(body)


def word_count(body: str) -> int:
    return len(words_list(body))


def character_count(body: str) -> int:
    return len(body)


def sentence_count(body: str) -> int:
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
    return len(sentences)


def _count_syllables(word: str) -> int:
    """Vowel-group heuristic: count contiguous runs of a/e/i/o/u/y, then
    drop one for a silent trailing 'e'. Not linguistically exact, but
    deterministic and documented, which is what reproducible FK scoring
    needs (§5.4).
    """
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 0
    syllables = 0
    prev_was_vowel = False
    for ch in w:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_was_vowel:
            syllables += 1
        prev_was_vowel = is_vowel
    if w.endswith("e") and syllables > 1:
        syllables -= 1
    return max(syllables, 1)


def fk_grade(body: str) -> float:
    """Flesch-Kincaid grade level, computed locally (no textstat dependency).

    FK Grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    Syllables use the vowel-group heuristic in `_count_syllables`.
    """
    words = words_list(body)
    n_words = len(words)
    n_sentences = sentence_count(body)
    if n_words == 0 or n_sentences == 0:
        raise ValueError("cannot compute FK grade for empty text")
    n_syllables = sum(_count_syllables(w) for w in words)
    return 0.39 * (n_words / n_sentences) + 11.8 * (n_syllables / n_words) - 15.59


def equation_count(body: str) -> int:
    """Count text segments that look like an equation.

    Rule (documented, not a parser): split the body on sentence-ish
    boundaries (newline, '.', '!', '?'). A segment counts as an equation if
    it contains '=' AND at least one digit or arithmetic operator
    (+, -, *, /) elsewhere in the segment. Determinism matters more than
    linguistic completeness here.
    """
    segments = _SENTENCE_SPLIT_RE.split(body)
    count = 0
    for seg in segments:
        if "=" in seg and re.search(r"[0-9+\-*/]", seg.replace("=", "")):
            count += 1
    return count


def lexical_diversity(body: str) -> float:
    """Type-token ratio over lowercased alphabetic tokens."""
    tokens = _ALPHA_TOKEN_RE.findall(body.lower())
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def duration_seconds(body: str, wpm: float = 220.0) -> float:
    """§6.2 main specification: 60 * words / wpm."""
    return 60.0 * word_count(body) / wpm


def word_onsets(body: str, wpm: float = 220.0) -> List[float]:
    """Per-word onset times in seconds, §6.2 equation (4):

        onset_j = 60 * cumulative_words_before_j / wpm

    Words are counted in reading order across sentences (sentence
    boundaries do not reset the cumulative count; they only delimit where
    `sentence_count`/`fk_grade` place sentence breaks). Not consumed
    downstream in this slice -- implemented now so the Phase II handoff to
    TRIBE timing is ready.
    """
    words = words_list(body)
    return [60.0 * i / wpm for i in range(len(words))]


@dataclass(frozen=True)
class FeatureVector:
    words: int
    characters: int
    sentences: int
    fk_grade: float
    equation_count: int
    example_count: int
    lexical_diversity: float
    duration_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "words": self.words,
            "characters": self.characters,
            "sentences": self.sentences,
            "fk_grade": self.fk_grade,
            "equation_count": self.equation_count,
            "example_count": self.example_count,
            "lexical_diversity": self.lexical_diversity,
            "duration_seconds": self.duration_seconds,
        }


def extract_features(body: str, *, example_count: int) -> FeatureVector:
    """Build the §5.3 observable-property vector for one stimulus body.

    `example_count` is declared in the stimulus front matter (counting
    worked examples requires semantic judgment) rather than computed here;
    every other field is derived from `body`.
    """
    return FeatureVector(
        words=word_count(body),
        characters=character_count(body),
        sentences=sentence_count(body),
        fk_grade=fk_grade(body),
        equation_count=equation_count(body),
        example_count=example_count,
        lexical_diversity=lexical_diversity(body),
        duration_seconds=duration_seconds(body),
    )
