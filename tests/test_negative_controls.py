"""Negative-control generators and their provenance stamps (§10.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import negative_controls as NC
from src.generation.features import word_count

TEXT = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."


def test_temporal_shuffle_preserves_word_count_and_therefore_duration():
    shuffled, provenance = NC.temporally_shuffled(TEXT, seed=0)
    assert word_count(shuffled) == word_count(TEXT)
    assert provenance.is_negative_control
    assert provenance.control == "temporally_shuffled"
    assert provenance.detail["n_sentences"] == 4


def test_temporal_shuffle_actually_reorders():
    shuffled, _ = NC.temporally_shuffled(TEXT, seed=0)
    assert shuffled != TEXT


def test_temporal_shuffle_is_reproducible_under_a_seed():
    assert NC.temporally_shuffled(TEXT, seed=3)[0] == NC.temporally_shuffled(TEXT, seed=3)[0]


def test_temporal_shuffle_of_a_single_sentence_raises():
    with pytest.raises(ValueError, match=">= 2 sentences"):
        NC.temporally_shuffled("Only one sentence here.", seed=0)


def test_irrelevant_matched_picks_the_closest_qualifying_donor():
    # single-syllable fillers throughout, so the donors differ from the source
    # in length only and the FK caliper is not what decides the match
    source = " ".join(["word"] * 100) + "."
    near = " ".join(["term"] * 98) + "."
    far = " ".join(["term"] * 50) + "."
    donor, provenance = NC.irrelevant_matched(source, [far, near])
    assert donor == near
    assert provenance.detail["donor_index"] == 1
    assert provenance.control == "irrelevant_matched"


def test_irrelevant_matched_raises_when_nothing_is_within_the_length_caliper():
    source = " ".join(["word"] * 100) + "."
    with pytest.raises(ValueError, match="no donor matched"):
        NC.irrelevant_matched(source, [" ".join(["term"] * 20) + "."])


def test_irrelevant_matched_rejects_a_length_matched_but_harder_donor():
    """Length matching alone is not enough: a donor that reads several grades
    harder would confound the control with difficulty."""
    source = " ".join(["word"] * 100) + "."
    harder = " ".join(["complicated"] * 100) + "."
    with pytest.raises(ValueError, match="no donor matched"):
        NC.irrelevant_matched(source, [harder])


def test_irrelevant_matched_rejects_an_empty_corpus():
    with pytest.raises(ValueError, match="donor_corpus is empty"):
        NC.irrelevant_matched(TEXT, [])


def _parcel_frame(n_units: int = 3) -> pd.DataFrame:
    rows = []
    for unit in range(n_units):
        for condition in ("traditional", "ai_scaffolding"):
            for parcel_id in (1, 2):
                for t in range(2):
                    rows.append(
                        {
                            "unit_id": f"u{unit}",
                            "condition": condition,
                            "parcel_id": parcel_id,
                            "time_index": t,
                            "mean_bold": float(unit * 10 + parcel_id + t),
                        }
                    )
    return pd.DataFrame(rows)


def test_permute_predictions_reassigns_units_and_keeps_the_original():
    permuted, provenance = NC.permute_predictions(_parcel_frame(), seed=1)
    assert (permuted["unit_id"] != permuted["source_unit_id"]).any()
    assert sorted(permuted["unit_id"].unique()) == sorted(permuted["source_unit_id"].unique())
    assert provenance.control == "permute_predictions"
    assert permuted.attrs[NC.CONTROL_FLAG] is True


def test_permute_predictions_preserves_each_prediction_intact():
    frame = _parcel_frame()
    permuted, _ = NC.permute_predictions(frame, seed=1)
    assert sorted(permuted["mean_bold"]) == sorted(frame["mean_bold"])


def test_permute_predictions_needs_two_units():
    with pytest.raises(ValueError, match=">= 2 units"):
        NC.permute_predictions(_parcel_frame(n_units=1), seed=0)


def test_permute_condition_labels_stays_within_each_unit():
    frame = _parcel_frame()
    permuted, provenance = NC.permute_condition_labels(frame, seed=2)
    for unit, block in permuted.groupby("unit_id"):
        original = sorted(frame[frame["unit_id"] == unit]["condition"])
        assert sorted(block["condition"]) == original
    assert set(provenance.detail["mappings"]) == {"u0", "u1", "u2"}


def test_permute_condition_labels_needs_two_conditions_per_unit():
    frame = _parcel_frame()
    frame = frame[~((frame["unit_id"] == "u1") & (frame["condition"] == "ai_scaffolding"))]
    with pytest.raises(ValueError, match="needs at least 2"):
        NC.permute_condition_labels(frame, seed=0)


def test_every_control_is_stamped_so_it_cannot_pass_as_substantive_output():
    permuted, provenance = NC.permute_predictions(_parcel_frame(), seed=0)
    record = permuted.attrs["negative_control"]
    assert record[NC.CONTROL_FLAG] is True
    assert record["control"] in NC.CONTROLS
    assert record["created_utc"]
    assert provenance.as_dict()[NC.CONTROL_FLAG] is True


def test_unknown_control_name_is_rejected():
    with pytest.raises(ValueError, match="unknown negative control"):
        NC.ControlProvenance(control="wishful_thinking", seed=0, source="x", detail={})


def test_permute_predictions_mapping_is_reproducible():
    first, _ = NC.permute_predictions(_parcel_frame(), seed=9)
    second, _ = NC.permute_predictions(_parcel_frame(), seed=9)
    assert np.array_equal(first["unit_id"].to_numpy(), second["unit_id"].to_numpy())
