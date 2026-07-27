"""Cache resolution, key derivation and manifest integrity (§4.5)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.tribe.cache import (
    MISS_NO_ENTRY,
    MISS_PARCEL_ONLY_VERTEX_REQUIRED,
    CacheError,
    TribeCache,
    keys_for,
    parcel_cache_key,
    text_hash,
)

TEXT = "Break-even analysis answers a simple question."


def _predictions(t=5, v=6):
    return np.arange(t * v, dtype=np.float32).reshape(t, v)


def _parcel_frame(stimulus_id="s1"):
    return pd.DataFrame(
        {
            "stimulus_id": stimulus_id,
            "time_index": [0, 0, 1, 1],
            "parcel_id": [1, 2, 1, 2],
            "network": ["A", "B", "A", "B"],
            "mean_bold": [1.0, 2.0, 3.0, 4.0],
            "sd_bold": [0.0, 0.0, 0.0, 0.0],
        }
    )


def _metadata(config, stimulus_id="s1", **extra):
    vertex_key, parcel_key = keys_for(TEXT, config)
    base = {
        "stimulus_id": stimulus_id,
        "text_hash": text_hash(TEXT),
        "reading_rate_wpm": config.reading_rate_wpm,
        "checkpoint": config.checkpoint,
        "revision": config.checkpoint_revision,
        "precision": config.precision,
        "atlas": config.atlas,
        "parcel_weighting": config.parcel_weighting,
        "n_timesteps": 5,
        "n_vertices": 6,
        "runtime_s": 1.0,
        "vertex_key": vertex_key,
        "parcel_key": parcel_key,
    }
    base.update(extra)
    return base


# -- branch 3: nothing cached ------------------------------------------------


def test_resolve_misses_when_nothing_is_cached(tribe_config):
    cache = TribeCache(tribe_config)
    result = cache.resolve("s1", TEXT, need_vertex=False)
    assert result.status == "miss"
    assert result.reason == MISS_NO_ENTRY
    assert result.needs_inference


# -- branch 1: parcel hit ----------------------------------------------------


def test_parcel_entry_is_a_hit_when_vertices_are_not_needed(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))

    result = cache.resolve("s1", TEXT, need_vertex=False)
    assert result.status == "hit_parcel"
    assert result.reason is None
    assert len(result.data) == 4


# -- the branch that matters: parcel-only, vertex required -------------------


def test_parcel_only_entry_is_a_miss_when_vertices_are_required(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))

    result = cache.resolve("s1", TEXT, need_vertex=True)
    assert result.status == "miss"
    assert result.reason == MISS_PARCEL_ONLY_VERTEX_REQUIRED
    assert result.data is None, "vertex data must never be fabricated from parcel means"


# -- branch 2: vertex hit ----------------------------------------------------


def test_vertex_entry_is_a_hit_for_both_kinds_of_request(tribe_config):
    cache = TribeCache(tribe_config)
    vertex_key, _ = keys_for(TEXT, tribe_config)
    cache.write_vertex(vertex_key, _predictions(), _metadata(tribe_config))

    for need_vertex in (True, False):
        result = cache.resolve("s1", TEXT, need_vertex=need_vertex)
        assert result.status == "hit_vertex"
        assert result.data.shape == (5, 6)


def test_vertex_round_trip_preserves_values_and_metadata(tribe_config):
    cache = TribeCache(tribe_config)
    vertex_key, _ = keys_for(TEXT, tribe_config)
    predictions = _predictions()
    cache.write_vertex(vertex_key, predictions, _metadata(tribe_config))

    restored, meta = cache.read_vertex(vertex_key)
    assert np.array_equal(restored, predictions)
    assert meta["stimulus_id"] == "s1"
    assert meta["revision"] == tribe_config.checkpoint_revision


def test_write_vertex_rejects_a_static_summary(tribe_config):
    cache = TribeCache(tribe_config)
    vertex_key, _ = keys_for(TEXT, tribe_config)
    with pytest.raises(CacheError, match="complete time-resolved matrix"):
        cache.write_vertex(vertex_key, np.zeros(6, dtype=np.float32), _metadata(tribe_config))


# -- key derivation ----------------------------------------------------------


def test_vertex_key_changes_with_reading_rate(tribe_config):
    before, _ = keys_for(TEXT, tribe_config)
    after, _ = keys_for(TEXT, replace(tribe_config, reading_rate_wpm=180.0))
    assert before != after


def test_vertex_key_changes_with_revision(tribe_config):
    before, _ = keys_for(TEXT, tribe_config)
    after, _ = keys_for(TEXT, replace(tribe_config, checkpoint_revision="b" * 40))
    assert before != after


def test_vertex_key_changes_with_precision(tribe_config):
    before, _ = keys_for(TEXT, tribe_config)
    after, _ = keys_for(TEXT, replace(tribe_config, precision="fp16"))
    assert before != after


def test_vertex_key_changes_with_text(tribe_config):
    before, _ = keys_for(TEXT, tribe_config)
    after, _ = keys_for(TEXT + " extra", tribe_config)
    assert before != after


def test_parcel_key_changes_with_atlas_and_weighting(tribe_config):
    vertex_key, parcel_key = keys_for(TEXT, tribe_config)

    other_weighting = parcel_cache_key(
        vertex_key=vertex_key, atlas=tribe_config.atlas, parcel_weighting="equal"
    )
    other_atlas = parcel_cache_key(
        vertex_key=vertex_key, atlas="something_else", parcel_weighting=tribe_config.parcel_weighting
    )
    assert parcel_key != other_weighting
    assert parcel_key != other_atlas


def test_parcel_key_does_not_change_the_vertex_key(tribe_config):
    """Parcel settings nest *under* the vertex prediction; they must not
    invalidate it, or changing the atlas would force a GPU rerun."""
    vertex_before, _ = keys_for(TEXT, tribe_config)
    vertex_after, _ = keys_for(TEXT, replace(tribe_config, parcel_weighting="equal"))
    assert vertex_before == vertex_after


def test_text_hash_is_newline_normalised():
    assert text_hash("a\r\nb") == text_hash("a\nb")


# -- manifest ----------------------------------------------------------------


def test_manifest_round_trip(tribe_config):
    cache = TribeCache(tribe_config)
    vertex_key, parcel_key = keys_for(TEXT, tribe_config)
    cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))
    cache.write_vertex(vertex_key, _predictions(), _metadata(tribe_config))

    manifest = cache.manifest()
    assert len(manifest) == 2
    assert set(manifest["level"]) == {"parcel", "vertex"}
    assert set(manifest["stimulus_id"]) == {"s1"}
    assert manifest["revision"].tolist() == [tribe_config.checkpoint_revision] * 2
    assert (manifest["size_bytes"] > 0).all()
    assert manifest["vertex_key"].tolist() == [vertex_key, vertex_key]


def test_manifest_row_requires_a_stimulus_id(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    metadata = _metadata(tribe_config)
    del metadata["stimulus_id"]
    with pytest.raises(CacheError, match="unattributable"):
        cache.write_parcel(parcel_key, _parcel_frame(), metadata)


def test_verify_manifest_is_clean_after_normal_writes(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))

    report = cache.verify_manifest()
    assert report == {"missing": [], "orphans": []}


def test_verify_manifest_detects_an_orphan(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))

    orphan = cache.parcel_path("f" * 64)
    orphan.parent.mkdir(parents=True, exist_ok=True)
    _parcel_frame().to_parquet(orphan, index=False)

    report = cache.verify_manifest()
    assert report["missing"] == []
    assert len(report["orphans"]) == 1
    assert report["orphans"][0].endswith(f"{'f' * 64}.parquet")


def test_verify_manifest_detects_a_missing_file(tribe_config):
    cache = TribeCache(tribe_config)
    _, parcel_key = keys_for(TEXT, tribe_config)
    path = cache.write_parcel(parcel_key, _parcel_frame(), _metadata(tribe_config))
    path.unlink()

    report = cache.verify_manifest()
    assert len(report["missing"]) == 1
    assert report["orphans"] == []
