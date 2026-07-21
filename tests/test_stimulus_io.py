"""Verification gate D: front matter parses cleanly and strips out of the body."""

import pytest

from src.generation.stimulus_io import parse_front_matter

STIMULUS_FILES = {
    "stimuli/traditional/be_001_primary.txt": "traditional",
    "stimuli/ai_scaffolding/be_001_primary.txt": "ai_scaffolding",
    "stimuli/ai_substitution/be_001_primary.txt": "ai_substitution",
}


@pytest.mark.parametrize("path,condition", STIMULUS_FILES.items())
def test_front_matter_parses_and_strips(path, condition):
    meta, body = parse_front_matter(path)
    assert meta["unit_id"] == "be_001"
    assert meta["condition"] == condition
    assert meta["variant"] == "primary"
    assert isinstance(meta["example_count"], int)
    assert isinstance(meta["learner_error_scripted"], bool)
    assert "---" not in body


@pytest.mark.parametrize("path,condition", STIMULUS_FILES.items())
def test_body_within_15_percent_of_430_words(path, condition):
    _, body = parse_front_matter(path)
    n_words = len(body.split())
    assert 430 * 0.85 <= n_words <= 430 * 1.15, f"{path}: {n_words} words out of range"


def test_parse_front_matter_raises_on_missing_delimiter(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("no front matter here", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_front_matter(bad_file)


def test_parse_front_matter_raises_on_unclosed_block(tmp_path):
    bad_file = tmp_path / "bad.txt"
    bad_file.write_text("---\nkey: value\nbody text with no closing delimiter", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_front_matter(bad_file)
