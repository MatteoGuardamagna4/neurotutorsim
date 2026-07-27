"""Config validation: unknown keys and malformed values must raise (§4.1)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from src.tribe.config import UNRESOLVED_REVISION, ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
SHIPPED_CONFIG = REPO_ROOT / "config" / "tribe.yaml"


def _write(tmp_path: Path, **overrides) -> Path:
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    raw.update(overrides)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "tribe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return path


def test_shipped_config_loads():
    config = load_config(SHIPPED_CONFIG)
    assert config.checkpoint == "facebook/tribev2"
    assert config.reading_rate_wpm == 220.0
    assert config.atlas == "schaefer400"
    assert config.project_root == REPO_ROOT


def test_shipped_config_ships_an_unresolved_revision():
    """The repo cannot pin a real SHA until Llama-3.2-3B access is approved.
    The placeholder must load for Track B and refuse for Track A."""
    config = load_config(SHIPPED_CONFIG)
    assert config.checkpoint_revision == UNRESOLVED_REVISION
    assert not config.revision_is_resolved
    with pytest.raises(ConfigError, match="unresolved"):
        config.require_resolved_revision()


def test_resolved_revision_is_returned(tmp_path):
    config = load_config(_write(tmp_path, checkpoint_revision="a" * 40))
    assert config.revision_is_resolved
    assert config.require_resolved_revision() == "a" * 40


def test_unknown_top_level_key_raises(tmp_path):
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, gpu_hours=12))


def test_unknown_metrics_key_raises(tmp_path):
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    raw["metrics"]["hrf_model"] = "spm"
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(_write(tmp_path, metrics=raw["metrics"]))


@pytest.mark.parametrize("revision", ["main", "v2", "A" * 40, "a" * 39, "z" * 40, ""])
def test_malformed_revision_raises(tmp_path, revision):
    with pytest.raises(ConfigError, match="40-character lowercase hex"):
        load_config(_write(tmp_path, checkpoint_revision=revision))


def test_unknown_precision_raises(tmp_path):
    with pytest.raises(ConfigError, match="precision must be one of"):
        load_config(_write(tmp_path, precision="bf16"))


def test_unsupported_reading_rate_raises(tmp_path):
    with pytest.raises(ConfigError, match="reading_rate_wpm must be one of"):
        load_config(_write(tmp_path, reading_rate_wpm=200.0))


@pytest.mark.parametrize("rate", [180.0, 220.0, 260.0])
def test_the_three_permitted_reading_rates_load(tmp_path, rate):
    assert load_config(_write(tmp_path, reading_rate_wpm=rate)).reading_rate_wpm == rate


def test_unknown_weighting_raises(tmp_path):
    with pytest.raises(ConfigError, match="parcel_weighting must be one of"):
        load_config(_write(tmp_path, parcel_weighting="inverse_variance"))


def test_missing_required_key_raises(tmp_path):
    raw = yaml.safe_load(SHIPPED_CONFIG.read_text(encoding="utf-8"))
    del raw["atlas"]
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    path = config_dir / "tribe.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="missing required key 'atlas'"):
        load_config(path)


def test_cache_root_override(tmp_path):
    config = load_config(SHIPPED_CONFIG, cache_root=str(tmp_path / "drive"))
    assert config.cache_root == (tmp_path / "drive").resolve()


def test_relative_paths_resolve_against_the_project_root():
    config = load_config(SHIPPED_CONFIG)
    assert config.vertex_retention_set == (REPO_ROOT / "config" / "vertex_retention_set.txt")
    assert config.checkpoints_lock == (REPO_ROOT / "config" / "checkpoints.lock")


def test_identity_covers_every_cache_key_component():
    identity = load_config(SHIPPED_CONFIG).identity()
    assert set(identity) == {
        "checkpoint", "checkpoint_revision", "precision", "reading_rate_wpm",
        "atlas", "parcel_weighting",
    }


def test_vertex_retention_set_is_read_and_comments_ignored(tmp_path):
    retention = tmp_path / "retain.txt"
    retention.write_text(
        textwrap.dedent(
            """\
            # a comment
            be_001_traditional_primary

            be_001_ai_scaffolding_primary   # trailing comment
            """
        ),
        encoding="utf-8",
    )
    config = load_config(_write(tmp_path, vertex_retention_set=str(retention)))
    assert config.read_vertex_retention_set() == (
        "be_001_traditional_primary",
        "be_001_ai_scaffolding_primary",
    )


def test_missing_retention_file_raises_rather_than_retaining_nothing(tmp_path):
    config = load_config(_write(tmp_path, vertex_retention_set=str(tmp_path / "absent.txt")))
    with pytest.raises(ConfigError, match="cannot be reconstructed"):
        config.read_vertex_retention_set()


def test_shipped_retention_set_matches_the_current_corpus():
    config = load_config(SHIPPED_CONFIG)
    assert set(config.read_vertex_retention_set()) == {
        "be_001_traditional_primary",
        "be_001_ai_scaffolding_primary",
        "be_001_ai_substitution_primary",
    }
