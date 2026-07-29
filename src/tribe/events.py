"""Canonical word-level event table (§6.2), and what it does *not* apply to.

The deterministic reading-rate formula is authoritative for how long a
simulated learner spends on a stimulus:

    onset_j = 60 * cumulative_words_before_j / r        (§6.2 eq. 4)

with r = 220 wpm in the main specification and r in {180, 260} available as
robustness values. That is a model of silent reading, and it governs Track B.

**It is not applied to TRIBE.** TRIBE renders the stimulus to speech and
encodes the resulting waveform, so its timings are a property of audio we do
not control. Two facts, both established against real vendor output, make an
override impossible rather than merely undesirable:

* its whisperx re-transcription tokenises differently from ours ('break-even'
  vs 'break' + 'even'), so no 1:1 word mapping exists to carry onsets across;
* its events frame is chunked -- one 'Audio' row per ~60 s segment, word events
  repeated per chunk, `start` relative to a chunk `offset` -- so absolute
  reading-rate onsets written into `start` describe audio that is not there.

The gap between the two is measured and reported, never reconciled: see
`measure_timing_discrepancy`. This reopens brief §6.2 for Track A only; the
formula's standing in Track B is unchanged.

Two layers, deliberately separated:

* `build_events` produces OUR canonical table. It owns the science.
* everything below the adapter banner reads TRIBE's schema. A vendor change
  touches only that section.

Onset computation itself is delegated to `src.generation.features.word_onsets`
-- this module never reimplements the formula.

Track A/B: no torch import, safe on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from src.generation.features import _SENTENCE_SPLIT_RE, _WORD_RE, word_onsets, words_list

#: Columns of the canonical event table.
EVENT_COLUMNS = ("word", "onset_s", "duration_s", "sentence_index", "word_index")

#: Column names TRIBE v2's events dataframe uses (as produced by
#: `TribeModel.get_events_dataframe`). Kept as constants so a vendor rename is
#: a one-line change here rather than a grep across the codebase.
TRIBE_COL_TYPE = "type"
TRIBE_COL_START = "start"
TRIBE_COL_DURATION = "duration"
TRIBE_COL_TEXT = "text"
TRIBE_COL_FILEPATH = "filepath"
TRIBE_COL_CONTEXT = "context"
#: Chunk start. Non-zero values mean `start` is relative to a ~60 s segment.
TRIBE_COL_OFFSET = "offset"

#: The vendor's own spelling of the event types it emits. TRIBE v2 capitalises
#: them ('Audio', 'Sentence', 'Text', 'Word'); matching is done
#: case-insensitively (see `_rows_of_type`) because getting the case wrong reads
#: as "this stimulus contains no words" rather than as a schema change.
TRIBE_WORD_TYPE = "Word"
TRIBE_AUDIO_TYPE = "Audio"


def build_events(stimulus_text: str, reading_rate_wpm: float) -> pd.DataFrame:
    """Canonical word-level event table for one stimulus body.

    Columns: word, onset_s, duration_s, sentence_index, word_index.

    * `onset_s` comes from `features.word_onsets` (§6.2 eq. 4) -- this function
      does not recompute it.
    * `duration_s` is the uniform per-word duration 60 / r implied by the same
      constant-rate assumption. Word onsets are therefore contiguous: word j
      ends exactly where word j+1 begins.
    * `sentence_index` preserves sentence boundaries, which the flat onset
      formula would otherwise erase. Sentences are delimited the same way
      `features.sentence_count` delimits them, so counts agree across modules.
    """
    if reading_rate_wpm <= 0:
        raise ValueError(f"reading_rate_wpm must be > 0; got {reading_rate_wpm}")

    words = words_list(stimulus_text)
    if not words:
        raise ValueError("stimulus text contains no words; cannot build an event table")

    onsets = word_onsets(stimulus_text, wpm=reading_rate_wpm)
    if len(onsets) != len(words):
        raise ValueError(
            f"word_onsets returned {len(onsets)} onsets for {len(words)} words -- "
            f"features.word_onsets and features.words_list have diverged"
        )

    sentence_index = _sentence_index_per_word(stimulus_text, words)
    per_word_duration = 60.0 / float(reading_rate_wpm)

    events = pd.DataFrame(
        {
            "word": words,
            "onset_s": onsets,
            "duration_s": [per_word_duration] * len(words),
            "sentence_index": sentence_index,
            "word_index": list(range(len(words))),
        }
    )
    validate_events(events)
    return events


def _sentence_index_per_word(body: str, words: Sequence[str]) -> List[int]:
    """Map each word (in reading order) to the index of its sentence.

    Uses the same splitter as `features.sentence_count`. Because the sentence
    delimiters ([.!?\\n]) are not word characters, splitting can never split or
    merge a word -- an invariant this function asserts rather than assumes.
    """
    indices: List[int] = []
    sentence_i = -1
    for segment in _SENTENCE_SPLIT_RE.split(body):
        if not segment.strip():
            continue
        sentence_i += 1
        indices.extend([sentence_i] * len(_WORD_RE.findall(segment)))

    if len(indices) != len(words):
        raise ValueError(
            f"sentence segmentation recovered {len(indices)} words but the body "
            f"tokenizes to {len(words)}; sentence boundaries cannot be assigned "
            f"without dropping or duplicating a word"
        )
    return indices


def validate_events(events: pd.DataFrame) -> None:
    """Assert the invariants every downstream consumer relies on."""
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"event table is missing column(s) {missing}")
    if events.empty:
        raise ValueError("event table is empty")

    onsets = events["onset_s"].to_numpy()
    if not (onsets[1:] >= onsets[:-1]).all():
        raise ValueError("event onsets are not monotonically increasing")
    if (events["duration_s"].to_numpy() < 0).any():
        raise ValueError("event table contains a negative duration")
    if (onsets < 0).any():
        raise ValueError("event table contains a negative onset")
    if list(events["word_index"]) != list(range(len(events))):
        raise ValueError("word_index must be a contiguous 0-based reading-order index")
    sentences = events["sentence_index"].to_numpy()
    if not (sentences[1:] >= sentences[:-1]).all():
        raise ValueError("sentence_index must be non-decreasing in reading order")


def total_duration_s(events: pd.DataFrame) -> float:
    """End time of the last word: the stimulus duration implied by the events."""
    validate_events(events)
    last = events.iloc[-1]
    return float(last["onset_s"] + last["duration_s"])


# ---------------------------------------------------------------------------
# TRIBE adapter. Everything below knows about the vendor's schema; nothing
# above does.
#
# NOTE: this section deliberately contains no way to write our onsets into a
# TRIBE events frame. See `measure_timing_discrepancy` for why, and do not add
# one back.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingDiscrepancy:
    """How far our reading-rate model sits from the audio TRIBE actually heard.

    Measured at the level the data supports -- counts and total duration -- not
    per word. TRIBE's word events cannot be aligned 1:1 with ours; see
    `measure_timing_discrepancy` for why.
    """

    stimulus_id: str
    reading_rate_wpm: float
    our_n_words: int
    our_total_duration_s: float
    tribe_n_word_events: int
    tribe_n_audio_rows: int
    tribe_span_s: float
    tribe_max_offset_s: float

    @property
    def duration_ratio(self) -> float:
        """TRIBE's audio span over our implied reading duration."""
        if self.our_total_duration_s <= 0:
            raise ValueError(
                f"{self.stimulus_id}: our_total_duration_s is {self.our_total_duration_s}; "
                f"a duration ratio is undefined"
            )
        return self.tribe_span_s / self.our_total_duration_s

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "reading_rate_wpm": self.reading_rate_wpm,
            "our_n_words": self.our_n_words,
            "our_total_duration_s": self.our_total_duration_s,
            "tribe_n_word_events": self.tribe_n_word_events,
            "tribe_n_audio_rows": self.tribe_n_audio_rows,
            "tribe_span_s": self.tribe_span_s,
            "tribe_max_offset_s": self.tribe_max_offset_s,
            "duration_ratio": self.duration_ratio,
        }


def _rows_of_type(tribe_df: pd.DataFrame, type_name: str) -> pd.DataFrame:
    """Rows of one TRIBE event type, matched case-insensitively.

    The vendor emits capitalised names ('Audio', 'Sentence', 'Text', 'Word').
    Matching case-sensitively on 'word' once made a frame full of words look
    like a frame containing none.
    """
    if TRIBE_COL_TYPE not in tribe_df.columns:
        return tribe_df.iloc[0:0]
    types = tribe_df[TRIBE_COL_TYPE].astype(str).str.strip().str.casefold()
    return tribe_df[(types == type_name.casefold()).to_numpy()]


def _finite_max(frame: pd.DataFrame, *columns: str) -> float:
    """Largest finite value of the row-wise sum of `columns`, or 0.0 if none.

    Non-numeric and missing entries count as zero rather than poisoning the
    maximum: TRIBE's frame is a union of row types, so columns that apply to one
    type are NaN for the others by design.
    """
    present = [c for c in columns if c in frame.columns]
    if not present or frame.empty:
        return 0.0
    total = None
    for column in present:
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        total = values if total is None else total + values
    finite = total[total.apply(lambda v: v == v and abs(v) != float("inf"))]
    return float(finite.max()) if not finite.empty else 0.0


def measure_timing_discrepancy(
    tribe_df: pd.DataFrame, events_df: pd.DataFrame, *, stimulus_id: str
) -> TimingDiscrepancy:
    """Record how far our reading-rate model sits from TRIBE's audio (§6.2).

    **This measures. It does not reconcile, and it does not overwrite.** The
    §6.2 formula stays authoritative for how long a simulated learner spends on
    a stimulus (Track B). It is not imposed on TRIBE, for two reasons
    established against real vendor output:

    * *The tokenisations cannot be aligned.* TRIBE renders the text to speech
      and re-transcribes it with whisperx, so it yields 'break-even' where we
      yield 'break' + 'even', and '120,000' + 'euros' where we yield 'EUR' +
      '120' + '000'. Two tokenisers over two media do not agree, by
      construction -- a 1:1 word mapping is unachievable in principle, not
      merely unwritten.
    * *TRIBE consumes the waveform.* The audio runs at TTS rate whatever an
      events table claims. The frame is chunked: one 'Audio' row per ~60 s
      segment, the word events repeated once per chunk, and an `offset` column
      carrying the chunk start. Writing absolute reading-rate onsets into
      `start` made the event timings describe audio that was not there.

    The quantities here are counts and spans, which are defined without any
    alignment. `tribe_span_s` is reported next to `tribe_max_offset_s` because
    the vendor's absolute-vs-chunk-relative convention is undocumented: when
    `tribe_max_offset_s` is non-zero, treat the span as a lower bound rather
    than trusting one interpretation silently.
    """
    validate_events(events_df)
    missing = sorted({TRIBE_COL_START, TRIBE_COL_DURATION} - set(tribe_df.columns))
    if missing:
        raise ValueError(
            f"TRIBE events dataframe is missing column(s) {missing}. Observed columns: "
            f"{sorted(tribe_df.columns)}. The vendor schema has changed -- update the "
            f"TRIBE_COL_* constants in src/tribe/events.py rather than working around it."
        )

    words = _rows_of_type(tribe_df, TRIBE_WORD_TYPE)
    if words.empty:
        raise ValueError(
            f"{stimulus_id}: TRIBE events dataframe has no rows of type "
            f"{TRIBE_WORD_TYPE!r} (compared case-insensitively); observed types: "
            f"{sorted(tribe_df[TRIBE_COL_TYPE].astype(str).unique())}"
            if TRIBE_COL_TYPE in tribe_df.columns
            else f"{stimulus_id}: TRIBE events dataframe has no {TRIBE_COL_TYPE!r} column"
        )

    return TimingDiscrepancy(
        stimulus_id=stimulus_id,
        reading_rate_wpm=float(60.0 / float(events_df["duration_s"].iloc[0])),
        our_n_words=int(len(events_df)),
        our_total_duration_s=total_duration_s(events_df),
        tribe_n_word_events=int(len(words)),
        tribe_n_audio_rows=int(len(_rows_of_type(tribe_df, TRIBE_AUDIO_TYPE))),
        tribe_span_s=_finite_max(tribe_df, TRIBE_COL_START, TRIBE_COL_DURATION),
        tribe_max_offset_s=_finite_max(tribe_df, TRIBE_COL_OFFSET),
    )


def write_timing_discrepancy_report(
    discrepancies: Sequence[TimingDiscrepancy], out_path: str | Path
) -> Path:
    """Write `reports/timing_discrepancy.md` (§3 of the Phase II task spec).

    Written for review, not acted upon. Under the settled §6.2 policy neither
    source is adjusted to match the other; this file is where the gap is stated
    plainly instead.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Timing: our reading-rate model vs the audio TRIBE heard",
        "",
        "The deterministic reading-rate formula (§6.2 eq. 4) is authoritative for how long",
        "a simulated learner spends on a stimulus. It is **not** applied to TRIBE, which",
        "renders the text to speech and encodes the resulting waveform: the audio runs at",
        "TTS rate whatever an events table says, and TRIBE's own word events cannot be",
        "aligned 1:1 with ours -- different tokenisers over different media, and the frame",
        "is chunked with the word events repeated once per chunk. Neither source is",
        "adjusted to match the other; the gap is recorded here.",
        "",
        "`TRIBE span` is the largest finite `start + duration` in the frame. Where",
        "`max offset` is non-zero the vendor is reporting chunk-relative times and the",
        "span is a lower bound on the true audio duration.",
        "",
        "| stimulus_id | r (wpm) | our words | our duration (s) | TRIBE word events "
        "| TRIBE audio rows | TRIBE span (s) | max offset (s) | span / our duration |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for d in discrepancies:
        lines.append(
            f"| {d.stimulus_id} | {d.reading_rate_wpm:.0f} | {d.our_n_words} | "
            f"{d.our_total_duration_s:.2f} | {d.tribe_n_word_events} | "
            f"{d.tribe_n_audio_rows} | {d.tribe_span_s:.2f} | "
            f"{d.tribe_max_offset_s:.2f} | {d.duration_ratio:.2f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
