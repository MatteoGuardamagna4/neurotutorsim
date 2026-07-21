"""Verification gate E: FK grade, duration, and word-onset checks."""

import pytest

from src.generation import features as F
from src.generation.stimulus_io import parse_front_matter

STIMULUS_PATHS = [
    "stimuli/traditional/be_001_primary.txt",
    "stimuli/ai_scaffolding/be_001_primary.txt",
    "stimuli/ai_substitution/be_001_primary.txt",
]


def test_fk_grade_known_string():
    # "Dog." -> 1 word, 1 sentence, 1 syllable.
    # 0.39*(1/1) + 11.8*(1/1) - 15.59 = -3.40
    assert F.fk_grade("Dog.") == pytest.approx(-3.40, abs=1e-9)


def test_duration_known_word_count():
    body = " ".join(["word"] * 220)
    assert F.duration_seconds(body) == pytest.approx(60.0)


def test_fk_grade_raises_on_empty_text():
    with pytest.raises(ValueError):
        F.fk_grade("")


@pytest.mark.parametrize("path", STIMULUS_PATHS)
def test_word_onsets_start_at_zero_and_nondecreasing(path):
    _, body = parse_front_matter(path)
    onsets = F.word_onsets(body)
    assert onsets[0] == 0.0
    assert all(b >= a for a, b in zip(onsets, onsets[1:]))


@pytest.mark.parametrize("path", STIMULUS_PATHS)
def test_extract_features_from_real_stimuli(path):
    meta, body = parse_front_matter(path)
    fv = F.extract_features(body, example_count=meta["example_count"])
    assert fv.words > 0
    assert fv.sentences > 0
    assert fv.duration_seconds == pytest.approx(60.0 * fv.words / 220.0)
