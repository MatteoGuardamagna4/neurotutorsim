"""Word onsets, monotonicity and sentence boundaries (§6.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src.tribe.events import (
    build_events,
    measure_timing_discrepancy,
    total_duration_s,
    validate_events,
    write_timing_discrepancy_report,
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


# -- the TRIBE adapter -------------------------------------------------------
#
# The frames below are shaped like real vendor output, which is the whole point:
# capitalised type names, four ~60 s Audio chunks, the word events repeated once
# per chunk, and a whisperx tokenisation that does not match ours.


def _vendor_frame(*, n_chunks=4, chunk_s=60.0, words=None):
    """A frame shaped like `TribeModel.get_events_dataframe()` really returns."""
    words = words or ["TUTOR.", "Let's", "work", "through", "a", "break-even"]
    rows = [{"type": "Text", "start": 0.0, "duration": 12.0, "text": TEXT, "offset": None}]
    for chunk in range(n_chunks):
        offset = chunk * chunk_s
        rows.append(
            {"type": "Audio", "start": 0.0, "duration": chunk_s, "text": None, "offset": offset}
        )
        rows += [
            {
                "type": "Word",
                "start": float(i) * 0.4,
                "duration": 0.4,
                "text": w,
                "offset": offset,
            }
            for i, w in enumerate(words)
        ]
    frame = pd.DataFrame(rows)
    # TRIBE concatenates one sub-frame per event type; the index repeats.
    frame.index = [0] * len(frame)
    return frame


def test_discrepancy_is_measured_without_aligning_words():
    """The vendor tokenisation disagrees with ours and the frame repeats itself
    four times over. Neither prevents a measurement."""
    events = build_events(TEXT, 220.0)
    frame = _vendor_frame()

    d = measure_timing_discrepancy(frame, events, stimulus_id="s1")

    assert d.our_n_words == 9
    assert d.tribe_n_word_events == 24  # 6 words x 4 chunks -- not 9, and that is fine
    assert d.tribe_n_audio_rows == 4
    assert d.tribe_max_offset_s == pytest.approx(180.0)
    assert d.our_total_duration_s == pytest.approx(60.0 * 9 / 220.0)
    assert d.reading_rate_wpm == pytest.approx(220.0)


def test_span_is_the_largest_finite_start_plus_duration():
    events = build_events(TEXT, 220.0)
    frame = _vendor_frame(n_chunks=1)
    # 6 words at 0.4 s steps -> last starts at 2.0, ends at 2.4; Audio row is 60 s.
    assert measure_timing_discrepancy(
        frame, events, stimulus_id="s1"
    ).tribe_span_s == pytest.approx(60.0)


def test_duration_ratio_reports_the_gap_between_the_two_models():
    events = build_events(TEXT, 220.0)
    d = measure_timing_discrepancy(_vendor_frame(), events, stimulus_id="s1")
    assert d.duration_ratio == pytest.approx(d.tribe_span_s / d.our_total_duration_s)


def test_a_frame_with_no_word_rows_names_the_observed_types():
    events = build_events(TEXT, 220.0)
    frame = pd.DataFrame(
        {
            "type": ["Audio", "Sentence"],
            "start": [0.0, 0.0],
            "duration": [1.0, 1.0],
            "text": [None, "One two three."],
        }
    )
    with pytest.raises(ValueError, match="Audio"):
        measure_timing_discrepancy(frame, events, stimulus_id="s1")


def test_lowercase_vendor_type_still_matches():
    """A recapitalisation upstream must not read as 'no words in this stimulus'
    -- the failure that cost a Colab session once."""
    events = build_events(TEXT, 220.0)
    frame = _vendor_frame(n_chunks=1)
    frame["type"] = frame["type"].str.lower()
    assert measure_timing_discrepancy(
        frame, events, stimulus_id="s1"
    ).tribe_n_word_events == 6


def test_missing_tribe_column_names_the_observed_schema():
    events = build_events(TEXT, 220.0)
    frame = _vendor_frame(n_chunks=1).drop(columns=["start"])
    with pytest.raises(ValueError, match="Observed columns"):
        measure_timing_discrepancy(frame, events, stimulus_id="s1")


def test_report_states_that_neither_source_was_adjusted(tmp_path):
    events = build_events(TEXT, 220.0)
    d = measure_timing_discrepancy(_vendor_frame(), events, stimulus_id="s1")

    out = write_timing_discrepancy_report([d], tmp_path / "timing_discrepancy.md")
    body = out.read_text(encoding="utf-8")

    assert "Neither source is" in body and "adjusted to match" in body
    assert "s1" in body
    # The reader must be able to see that the span is chunk-relative.
    assert "180.00" in body
