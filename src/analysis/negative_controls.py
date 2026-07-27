"""Negative-control generators (§10.3 hooks).

Generators only. Running them, and interpreting what they produce, is a later
task (decision gate 20: no substantive conclusions until specification curve
and negative controls are done).

A negative control is a stimulus or a dataset that *should* produce no effect.
Its whole value is that it is indistinguishable from real output except for its
provenance -- which is exactly what makes it dangerous. Every function here
therefore returns `(data, ControlProvenance)`, and the provenance record is
stamped into `DataFrame.attrs` as well, so a control can never be mistaken for
substantive output downstream.

Track B: numpy, pandas. No torch.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.generation.features import _SENTENCE_SPLIT_RE, fk_grade, word_count

#: Key stamped into `.attrs` and into every provenance record.
CONTROL_FLAG = "is_negative_control"

CONTROLS = (
    "temporally_shuffled",
    "irrelevant_matched",
    "permute_predictions",
    "permute_condition_labels",
)


@dataclass(frozen=True)
class ControlProvenance:
    """Marks an artifact as a negative control. Never optional, never dropped."""

    control: str
    seed: Optional[int]
    source: str
    detail: Dict[str, Any]
    created_utc: str = ""
    is_negative_control: bool = True

    def __post_init__(self) -> None:
        if self.control not in CONTROLS:
            raise ValueError(f"unknown negative control {self.control!r}; expected one of {list(CONTROLS)}")
        if not self.created_utc:
            object.__setattr__(self, "created_utc", datetime.now(timezone.utc).isoformat())

    def as_dict(self) -> Dict[str, Any]:
        return {
            CONTROL_FLAG: True,
            "control": self.control,
            "seed": self.seed,
            "source": self.source,
            "detail": self.detail,
            "created_utc": self.created_utc,
        }


def _stamp(frame: pd.DataFrame, provenance: ControlProvenance) -> pd.DataFrame:
    out = frame.copy()
    out.attrs = {**frame.attrs, "negative_control": provenance.as_dict(), CONTROL_FLAG: True}
    return out


def _split_sentences(text: str) -> List[str]:
    """Sentences with their terminal punctuation preserved.

    Uses the same delimiter set as `features.sentence_count`, so a shuffled
    stimulus has the same sentence count as its source under every existing
    diagnostic.
    """
    sentences: List[str] = []
    cursor = 0
    for match in _SENTENCE_SPLIT_RE.finditer(text):
        segment = text[cursor : match.end()]
        if segment.strip():
            sentences.append(segment)
        cursor = match.end()
    tail = text[cursor:]
    if tail.strip():
        sentences.append(tail)
    return sentences


def temporally_shuffled(text: str, seed: int) -> Tuple[str, ControlProvenance]:
    """Sentence-shuffled stimulus: same words, same duration, destroyed discourse.

    Word count is identical by construction, so the §6.2 duration
    `60 * words / r` is identical too. What changes is only the order in which
    sentences arrive -- if a condition effect survives this, it is not carried
    by discourse structure.

    Raises below two sentences: shuffling one sentence returns the original text
    and would be a control in name only.
    """
    sentences = _split_sentences(text)
    if len(sentences) < 2:
        raise ValueError(
            f"temporally_shuffled needs >= 2 sentences to shuffle; got {len(sentences)}. A "
            f"one-sentence 'shuffle' is the original stimulus with a control label on it."
        )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(sentences))
    if len(sentences) > 1 and np.array_equal(order, np.arange(len(sentences))):
        order = np.roll(order, 1)  # an identity permutation is not a shuffle
    shuffled = "".join(sentences[i].strip() + " " for i in order).strip()

    if word_count(shuffled) != word_count(text):
        raise ValueError(
            f"sentence shuffling changed the word count ({word_count(text)} -> "
            f"{word_count(shuffled)}); the control would no longer be duration-matched"
        )

    provenance = ControlProvenance(
        control="temporally_shuffled",
        seed=seed,
        source=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        detail={
            "n_sentences": len(sentences),
            "order": order.tolist(),
            "word_count": word_count(text),
        },
    )
    return shuffled, provenance


def irrelevant_matched(
    text: str,
    donor_corpus: Sequence[str],
    *,
    length_tolerance: float = 0.10,
    fk_tolerance: float = 1.0,
) -> Tuple[str, ControlProvenance]:
    """Pick a semantically unrelated donor text matched on length and readability.

    The donor is the candidate closest in word count that falls inside both
    calipers: within `length_tolerance` of the source word count (the §5.5
    caliper, 10% by default) and within `fk_tolerance` Flesch-Kincaid grades.

    Raises if no donor qualifies. An unmatched "irrelevant" stimulus is not a
    negative control -- any difference it produces is confounded with length or
    difficulty, which is the exact confound the control exists to rule out.

    Semantic unrelatedness is the caller's responsibility: this function matches
    on observable properties and cannot verify that a donor is about a different
    topic.
    """
    if not donor_corpus:
        raise ValueError("donor_corpus is empty; there is nothing to match against")

    target_words = word_count(text)
    target_fk = fk_grade(text)
    if target_words == 0:
        raise ValueError("source text has no words")

    candidates = []
    rejected = []
    for index, donor in enumerate(donor_corpus):
        donor_words = word_count(donor)
        if donor_words == 0:
            rejected.append((index, "donor has no words"))
            continue
        length_gap = abs(donor_words - target_words) / target_words
        fk_gap = abs(fk_grade(donor) - target_fk)
        if length_gap <= length_tolerance and fk_gap <= fk_tolerance:
            candidates.append((length_gap, fk_gap, index, donor))
        else:
            rejected.append(
                (index, f"length gap {length_gap:.1%}, FK gap {fk_gap:.2f}")
            )

    if not candidates:
        raise ValueError(
            f"no donor matched the source ({target_words} words, FK {target_fk:.2f}) within "
            f"{length_tolerance:.0%} length and {fk_tolerance} FK grades. Rejections: "
            f"{rejected[:5]}{'...' if len(rejected) > 5 else ''}"
        )

    length_gap, fk_gap, index, donor = min(candidates, key=lambda c: (c[0], c[1]))
    provenance = ControlProvenance(
        control="irrelevant_matched",
        seed=None,
        source=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        detail={
            "donor_index": index,
            "source_words": target_words,
            "donor_words": word_count(donor),
            "length_gap": length_gap,
            "source_fk": target_fk,
            "fk_gap": fk_gap,
            "n_candidates": len(candidates),
        },
    )
    return donor, provenance


def permute_predictions(
    parcel_df: pd.DataFrame, seed: int
) -> Tuple[pd.DataFrame, ControlProvenance]:
    """Randomly reassign whole TRIBE predictions across units.

    Permutes at the level of `(unit_id, condition)` stimuli, keeping each
    prediction's internal time and parcel structure intact -- only the unit it
    is attached to changes. A contrast that survives this is being driven by
    something other than the stimulus each unit actually received.

    Raises below two units: with one unit the only permutation is the identity.
    """
    required = {"unit_id", "condition", "parcel_id", "time_index"}
    missing = sorted(required - set(parcel_df.columns))
    if missing:
        raise ValueError(f"parcel table is missing column(s) {missing}")

    units = np.array(sorted(parcel_df["unit_id"].unique()))
    if units.size < 2:
        raise ValueError(
            f"permute_predictions needs >= 2 units to permute across; got {units.size}. "
            f"Permuting one unit onto itself is the identity, not a control."
        )

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(units)
    if np.array_equal(shuffled, units):
        shuffled = np.roll(shuffled, 1)
    mapping = dict(zip(units.tolist(), shuffled.tolist()))

    out = parcel_df.copy()
    out["unit_id"] = out["unit_id"].map(mapping)
    out["source_unit_id"] = parcel_df["unit_id"].to_numpy()

    provenance = ControlProvenance(
        control="permute_predictions",
        seed=seed,
        source="parcel_df",
        detail={"n_units": int(units.size), "mapping": mapping},
    )
    return _stamp(out, provenance), provenance


def permute_condition_labels(
    df: pd.DataFrame, seed: int, *, unit_column: str = "unit_id"
) -> Tuple[pd.DataFrame, ControlProvenance]:
    """Shuffle condition labels **within** each matched unit set.

    Never across units: units are matched on observable stimulus properties
    (§5.3), and a permutation that moves a label to a different unit tests a
    hypothesis about between-unit differences that nobody posed.

    Raises if any unit has fewer than two conditions to permute.
    """
    if "condition" not in df.columns or unit_column not in df.columns:
        raise ValueError(
            f"table needs a 'condition' column and a {unit_column!r} column; observed: "
            f"{sorted(df.columns)}"
        )

    rng = np.random.default_rng(seed)
    out = df.copy()
    out["source_condition"] = df["condition"].to_numpy()
    mappings: Dict[Any, Dict[str, str]] = {}

    for unit, block in df.groupby(unit_column, sort=True):
        conditions = np.array(sorted(block["condition"].unique()))
        if conditions.size < 2:
            raise ValueError(
                f"unit {unit!r} has {conditions.size} condition(s); within-unit permutation "
                f"needs at least 2. Refusing to permute across units to make up the shortfall."
            )
        permuted = rng.permutation(conditions)
        if np.array_equal(permuted, conditions):
            permuted = np.roll(permuted, 1)
        mapping = dict(zip(conditions.tolist(), permuted.tolist()))
        mappings[unit] = mapping
        mask = out[unit_column] == unit
        out.loc[mask, "condition"] = out.loc[mask, "source_condition"].map(mapping)

    provenance = ControlProvenance(
        control="permute_condition_labels",
        seed=seed,
        source="df",
        detail={"n_units": len(mappings), "mappings": {str(k): v for k, v in mappings.items()}},
    )
    return _stamp(out, provenance), provenance
