"""The Colab session bootstrap: paths, the storage policy, and re-runnability.

These run off Colab, which is the point -- `setup_session` only sets
environment variables and makes directories, so the thing three notebooks
depend on is testable on a laptop with no GPU and no network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.colab import SessionError, setup_session


@pytest.fixture
def roots(tmp_path: Path):
    """A fake Drive and a fake local disk, with nothing in either."""
    drive = tmp_path / "MyDrive"
    local = tmp_path / "content"
    drive.mkdir()
    local.mkdir()
    return drive, local


def _setup(roots, **overrides):
    drive, local = roots
    kwargs = dict(drive_root=drive, local_root=local, authenticate=False)
    kwargs.update(overrides)
    return setup_session(**kwargs)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def test_paths_are_derived_and_created(roots, monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    drive, local = roots

    paths = _setup(roots)

    assert paths.project_drive == drive / "NeuroTutorSim"
    assert paths.cache_root == drive / "NeuroTutorSim" / "tribe_cache"
    assert paths.handoff == drive / "NeuroTutorSim" / "gate17"
    assert paths.hf_cache == local / "hf_cache"
    for directory in (paths.project_drive, paths.cache_root, paths.handoff, paths.hf_cache):
        assert directory.is_dir()


def test_the_subprocess_environment_is_exported(roots, monkeypatch):
    """The `!` cells are subprocesses: they inherit os.environ, not variables."""
    monkeypatch.delenv("HF_HOME", raising=False)
    import os

    paths = _setup(roots)

    assert os.environ["HF_HOME"] == str(paths.hf_cache)
    assert os.environ["CACHE_ROOT"] == str(paths.cache_root)
    assert os.environ["REPO_DIR"] == str(paths.repo_dir)


def test_cache_root_can_be_redirected(roots, tmp_path, monkeypatch):
    """The one thing a second inference notebook needs to differ in."""
    monkeypatch.delenv("HF_HOME", raising=False)
    elsewhere = tmp_path / "somewhere_else"

    paths = _setup(roots, cache_root=elsewhere)

    assert paths.cache_root == elsewhere
    assert elsewhere.is_dir()
    # the handoff still comes from the project folder -- only outputs moved
    assert paths.handoff == roots[0] / "NeuroTutorSim" / "gate17"


# ---------------------------------------------------------------------------
# The storage policy
# ---------------------------------------------------------------------------


def test_the_hf_cache_is_never_placed_on_drive(roots, monkeypatch):
    """~13 GB of weights on a 15 GB Drive leaves no room for the corpus.

    Weights re-download in about a minute; predictions cost GPU time against a
    gated model. Putting the cheap thing on the scarce disk is the mistake this
    guard exists to prevent from creeping back.
    """
    monkeypatch.delenv("HF_HOME", raising=False)
    drive, _ = roots

    with pytest.raises(SessionError, match="inside the Drive root"):
        _setup(roots, local_root=drive / "hf_on_drive_by_mistake")


# ---------------------------------------------------------------------------
# Re-runnability
# ---------------------------------------------------------------------------


def _fake_hub(monkeypatch, cache_dir: Path):
    """An already-imported huggingface_hub whose cache is fixed at `cache_dir`."""
    constants = types.ModuleType("huggingface_hub.constants")
    constants.HF_HUB_CACHE = str(cache_dir)
    hub = types.ModuleType("huggingface_hub")
    hub.constants = constants
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)


def test_re_running_after_the_hub_was_imported_is_a_no_op(roots, monkeypatch):
    """This is the cell you re-run after a restart or a dropped connection."""
    monkeypatch.delenv("HF_HOME", raising=False)
    _, local = roots
    _fake_hub(monkeypatch, local / "hf_cache" / "hub")

    paths = _setup(roots)

    assert paths.hf_cache == local / "hf_cache"


def test_a_hub_already_pointed_elsewhere_raises(roots, tmp_path, monkeypatch):
    """HF_HOME is read once, at import. Setting it later is silently ignored."""
    monkeypatch.delenv("HF_HOME", raising=False)
    _fake_hub(monkeypatch, tmp_path / "wrong_place" / "hub")

    with pytest.raises(SessionError, match="already imported"):
        _setup(roots)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


def test_a_missing_token_off_colab_is_reported_not_raised(roots, monkeypatch):
    """Track B laptops never need a token; Track A fails later, on the gate."""
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    paths = _setup(roots, authenticate=True)

    assert "not authenticated" in paths.auth_status
