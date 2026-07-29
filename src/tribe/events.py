"""Canonical word-level event table for TRIBE (§6.2).

Timing is settled: the deterministic reading-rate formula is authoritative.

    onset_j = 60 * cumulative_words_before_j / r        (§6.2 eq. 4)

with r = 220 wpm in the main specification and r in {180, 260} available as
robustness values. Onsets are *not* derived from TTS or from whisperx
re-transcription. If TRIBE's own `get_events_dataframe()` disagrees, our onsets
win and the disagreement is logged (see `timing_discrepancy`), never silently
reconciled in either direction.

Two layers, deliberately separated:

* `build_events` produces OUR canonical table. It owns the science.
* `to_tribe_events` / `apply_our_timings` adapt that table to whatever column
  names TRIBE currently expects. They own the vendor API. A TRIBE change
  touches only the adapter.

Onset computation itself is delegated to `src.generation.features.word_onsets`
-- this module never reimplements the formula.

Track A/B: no torch import, safe on a laptop.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

#: The vendor's own spelling of the word-event type. TRIBE v2 emits capitalised
#: type names ('Audio', 'Sentence', 'Text', 'Word'); matching is done
#: case-insensitively (see `_tribe_word_mask`) so a recapitalisation upstream is
#: not a crash, but frames *we* construct use the vendor's spelling verbatim.
TRIBE_WORD_TYPE = "Word"


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
# ---------------------------------------------------------------------------


def to_tribe_events(
    events_df: pd.DataFrame,
    *,
    filepath: str | Path,
    context: Optional[str] = None,
) -> pd.DataFrame:
    """Adapt the canonical table to TRIBE v2's events dataframe schema.

    TRIBE consumes rows of (type, start, duration, filepath, text, context).
    `filepath` is the audio rendering of the stimulus that TRIBE's own
    `get_events_dataframe` produced -- we reuse its audio, only its timings are
    replaced.

    Adapter only: no science here, and no shape change to `events_df`.
    """
    validate_events(events_df)
    return pd.DataFrame(
        {
            TRIBE_COL_TYPE: TRIBE_WORD_TYPE,
            TRIBE_COL_START: events_df["onset_s"].astype(float).to_numpy(),
            TRIBE_COL_DURATION: events_df["duration_s"].astype(float).to_numpy(),
            TRIBE_COL_FILEPATH: str(filepath),
            TRIBE_COL_TEXT: events_df["word"].to_numpy(),
            TRIBE_COL_CONTEXT: context,
        }
    )


@dataclass(frozen=True)
class TimingDiscrepancy:
    """How far TRIBE's own word timings sit from our 220 wpm onsets."""

    stimulus_id: str
    reading_rate_wpm: float
    n_words: int
    max_abs_offset_s: float
    mean_abs_offset_s: float
    tribe_total_duration_s: float
    our_total_duration_s: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "stimulus_id": self.stimulus_id,
            "reading_rate_wpm": self.reading_rate_wpm,
            "n_words": self.n_words,
            "max_abs_offset_s": self.max_abs_offset_s,
            "mean_abs_offset_s": self.mean_abs_offset_s,
            "tribe_total_duration_s": self.tribe_total_duration_s,
            "our_total_duration_s": self.our_total_duration_s,
        }


def _tribe_word_mask(tribe_df: pd.DataFrame) -> "pd.Series[bool]":
    """Boolean mask selecting the word rows of a TRIBE events frame.

    A mask rather than a sub-frame, because `apply_our_timings` has to write
    corrected timings back into the *full* frame: TRIBE's other row types
    ('Audio', 'Sentence', 'Text') are what its audio and context pathways read,
    and `model.predict` is only ever handed the whole frame -- that is what gate
    17 verifies against Meta's example.

    Type matching is case-insensitive. The vendor emits 'Word'; a build that
    emitted 'word' would otherwise look like a frame containing no words at all.
    """
    required = {TRIBE_COL_START, TRIBE_COL_DURATION, TRIBE_COL_TEXT}
    missing = sorted(required - set(tribe_df.columns))
    if missing:
        raise ValueError(
            f"TRIBE events dataframe is missing column(s) {missing}. Observed columns: "
            f"{sorted(tribe_df.columns)}. The vendor schema has changed -- update the "
            f"TRIBE_COL_* constants in src/tribe/events.py rather than working around it."
        )
    if TRIBE_COL_TYPE not in tribe_df.columns:
        return pd.Series(True, index=tribe_df.index)

    types = tribe_df[TRIBE_COL_TYPE].astype(str).str.strip().str.casefold()
    mask = types == TRIBE_WORD_TYPE.casefold()
    if not mask.any():
        raise ValueError(
            f"TRIBE events dataframe has no rows of type {TRIBE_WORD_TYPE!r} "
            f"(compared case-insensitively); observed types: "
            f"{sorted(tribe_df[TRIBE_COL_TYPE].astype(str).unique())}"
        )
    return mask


def apply_our_timings(
    tribe_df: pd.DataFrame, events_df: pd.DataFrame, *, stimulus_id: str
) -> tuple[pd.DataFrame, TimingDiscrepancy]:
    """Overwrite TRIBE's word timings with our 220 wpm onsets (§6.2, settled).

    Returns the corrected TRIBE frame plus the measured discrepancy. Raises if
    the two word sequences do not align: a partial overwrite would silently
    attach our onsets to the wrong words, which is worse than a hard failure.

    The frame returned is the *whole* input frame with the word rows retimed --
    same rows, same order, same columns. Non-word rows ('Audio', 'Sentence',
    'Text') are passed through untouched: they index the real TTS waveform that
    TRIBE just rendered, so retiming them to our reading rate would desynchronise
    the frame from the audio it describes. The resulting gap between our word
    onsets and TRIBE's audio is the discrepancy this function measures; per §6.2
    it is recorded, not reconciled.
    """
    validate_events(events_df)
    mask = _tribe_word_mask(tribe_df)
    words = tribe_df[mask.to_numpy()]

    ours = [w.lower() for w in events_df["word"].tolist()]
    theirs = [_normalize_token(t) for t in words[TRIBE_COL_TEXT].tolist()]

    if len(ours) != len(theirs):
        raise ValueError(
            f"{stimulus_id}: TRIBE produced {len(theirs)} word events but our event "
            f"table has {len(ours)}. Word sequences must align 1:1 before timings can "
            f"be replaced; refusing to truncate or pad."
        )
    mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(ours, theirs)) if a != b]
    if mismatches:
        head = mismatches[:5]
        raise ValueError(
            f"{stimulus_id}: {len(mismatches)} word(s) differ between our event table "
            f"and TRIBE's transcription; first mismatches (index, ours, tribe): {head}"
        )

    their_starts = words[TRIBE_COL_START].astype(float).to_numpy()
    our_starts = events_df["onset_s"].astype(float).to_numpy()
    offsets = abs(their_starts - our_starts)

    # Positional, not label-based: TRIBE builds this frame by concatenating one
    # sub-frame per event type, so its index is not guaranteed to be unique and
    # `.loc` on a duplicated label would write to the wrong rows.
    corrected = tribe_df.reset_index(drop=True)
    positions = mask.to_numpy().nonzero()[0]
    corrected.iloc[positions, corrected.columns.get_loc(TRIBE_COL_START)] = our_starts
    corrected.iloc[positions, corrected.columns.get_loc(TRIBE_COL_DURATION)] = (
        events_df["duration_s"].astype(float).to_numpy()
    )

    their_last = words.iloc[-1]
    discrepancy = TimingDiscrepancy(
        stimulus_id=stimulus_id,
        reading_rate_wpm=float(60.0 / float(events_df["duration_s"].iloc[0])),
        n_words=len(ours),
        max_abs_offset_s=float(offsets.max()),
        mean_abs_offset_s=float(offsets.mean()),
        tribe_total_duration_s=float(
            their_last[TRIBE_COL_START] + their_last[TRIBE_COL_DURATION]
        ),
        our_total_duration_s=total_duration_s(events_df),
    )
    return corrected, discrepancy


def _normalize_token(token: Any) -> str:
    """Reduce a TRIBE transcript token to the same alphabet our tokenizer uses."""
    matches = _WORD_RE.findall(str(token))
    return "".join(matches).lower()


def write_timing_discrepancy_report(
    discrepancies: Sequence[TimingDiscrepancy], out_path: str | Path
) -> Path:
    """Write `reports/timing_discrepancy.md` (§3 of the Phase II task spec).

    Written for review, not acted upon: our onsets are already authoritative by
    the time this is called.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Timing discrepancy: 220 wpm onsets vs TRIBE's own event timings",
        "",
        "The deterministic reading-rate formula (§6.2 eq. 4) is authoritative for word",
        "onsets. TRIBE's `get_events_dataframe()` derives its own timings by running TTS",
        "and re-transcribing with whisperx. Where the two disagree, our onsets are used",
        "and the disagreement is recorded here. Neither source was adjusted to match the",
        "other.",
        "",
        "| stimulus_id | r (wpm) | n words | max abs offset (s) | mean abs offset (s) "
        "| TRIBE duration (s) | our duration (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for d in discrepancies:
        lines.append(
            f"| {d.stimulus_id} | {d.reading_rate_wpm:.0f} | {d.n_words} | "
            f"{d.max_abs_offset_s:.3f} | {d.mean_abs_offset_s:.3f} | "
            f"{d.tribe_total_duration_s:.2f} | {d.our_total_duration_s:.2f} |"
        )
    lines.append("")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out
