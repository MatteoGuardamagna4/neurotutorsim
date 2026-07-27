"""Word onsets, monotonicity and sentence boundaries (§6.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tribe.events import (
    apply_our_timings,
    build_events,
    to_tribe_events,
    total_duration_s,
    validate_events,
)

TEXT = "One two three. Four five! Six seven eight nine?"


@pytest.mark.parametrize(
    "wpm,expected_step",
    [
        (220.0, 60.0 / 220.0),
        (180.0, 60.0 / 180.0),
        (260.0, 60.0 / 260.0),
    ],
)
def test_onsets_match_hand_computed_values(wpm, expected_step):
    events = build_events(TEXT, wpm)
    assert len(events) == 9
    for j in range(9):
        # onset_j = 60 * cumulative_words_before_j / r
        assert events.loc[j, "onset_s"] == pytest.approx(60.0 * j / wpm)
    assert events["duration_s"].unique().tolist() == pytest.approx([expected_step])
    assert total_duration_s(events) == pytest.approx(60.0 * 9 / wpm)


def test_onsets_are_monotonic_and_word_index_is_contiguous():
    events = build_events(TEXT, 220.0)
    assert events["onset_s"].is_monotonic_increasing
    assert events["word_index"].tolist() == list(range(9))
    validate_events(events)  # must not raise


def test_sentence_boundaries_are_preserved():
    events = build_events(TEXT, 220.0)
    assert events["sentence_index"].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert events.groupby("sentence_index")["word"].count().tolist() == [3, 2, 4]


def test_sentence_index_survives_a_newline_delimiter():
    events = build_events("Alpha beta\ngamma delta.", 220.0)
    assert events["sentence_index"].tolist() == [0, 0, 1, 1]


def test_empty_text_raises():
    with pytest.raises(ValueError, match="no words"):
        build_events("   ", 220.0)


def test_non_positive_rate_raises():
    with pytest.raises(ValueError, match="reading_rate_wpm"):
        build_events(TEXT, 0.0)


def test_validate_events_rejects_non_monotonic_onsets():
    events = build_events(TEXT, 220.0)
    events.loc[3, "onset_s"] = 0.0
    with pytest.raises(ValueError, match="monotonically increasing"):
        validate_events(events)


def test_validate_events_rejects_negative_duration():
    events = build_events(TEXT, 220.0)
    events.loc[2, "duration_s"] = -1.0
    with pytest.raises(ValueError, match="negative duration"):
        validate_events(events)


def test_to_tribe_events_is_a_pure_adapter():
    events = build_events(TEXT, 220.0)
    adapted = to_tribe_events(events, filepath="/tmp/a.wav", context=None)
    assert len(adapted) == len(events)
    assert set(adapted.columns) == {"type", "start", "duration", "filepath", "text", "context"}
    assert adapted["start"].tolist() == pytest.approx(events["onset_s"].tolist())
    assert adapted["text"].tolist() == events["word"].tolist()


def _fake_tribe_frame(words, starts, durations):
    return pd.DataFrame(
        {
            "type": "word",
            "start": starts,
            "duration": durations,
            "filepath": "/tmp/a.wav",
            "text": words,
            "context": None,
        }
    )


def test_our_timings_win_and_the_discrepancy_is_measured():
    events = build_events(TEXT, 220.0)
    drifted = [float(o) + 0.5 for o in events["onset_s"]]
    tribe_frame = _fake_tribe_frame(events["word"].tolist(), drifted, [0.4] * 9)

    corrected, discrepancy = apply_our_timings(tribe_frame, events, stimulus_id="s1")

    assert corrected["start"].tolist() == pytest.approx(events["onset_s"].tolist())
    assert corrected["duration"].tolist() == pytest.approx(events["duration_s"].tolist())
    assert discrepancy.max_abs_offset_s == pytest.approx(0.5)
    assert discrepancy.mean_abs_offset_s == pytest.approx(0.5)
    assert discrepancy.n_words == 9


def test_misaligned_word_sequence_raises_rather_than_truncating():
    events = build_events(TEXT, 220.0)
    words = events["word"].tolist()[:-1]
    tribe_frame = _fake_tribe_frame(words, list(range(8)), [0.4] * 8)
    with pytest.raises(ValueError, match="align 1:1"):
        apply_our_timings(tribe_frame, events, stimulus_id="s1")


def test_mismatched_word_text_raises():
    events = build_events(TEXT, 220.0)
    words = events["word"].tolist()
    words[4] = "banana"
    tribe_frame = _fake_tribe_frame(words, list(range(9)), [0.4] * 9)
    with pytest.raises(ValueError, match="differ between our event table"):
        apply_our_timings(tribe_frame, events, stimulus_id="s1")


def test_missing_tribe_column_names_the_observed_schema():
    events = build_events(TEXT, 220.0)
    frame = _fake_tribe_frame(events["word"].tolist(), list(range(9)), [0.4] * 9)
    with pytest.raises(ValueError, match="Observed columns"):
        apply_our_timings(frame.drop(columns=["start"]), events, stimulus_id="s1")
