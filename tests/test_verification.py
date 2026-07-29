"""How `load_model` pins a checkpoint on builds that cannot be told a revision.

The published `tribev2` build takes `checkpoint_dir` and resolves the repo id
itself -- there is no `revision` argument to pass. Handing it the bare repo id
would load whatever `main` points at today while every cache key claims the
pinned SHA (§6.1). These tests fake `tribev2.demo_utils` so the dispatch can be
exercised on CPU with no network: what matters is *which* argument the pin
arrives in, and that an unpinnable build stops instead of loading.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from src.tribe import verification
from src.tribe.config import ConfigError
from src.tribe.verification import VerificationError

from tests.conftest import PINNED_REVISION


class _FakeTribeModel:
    """Records the call. `eval()` on the inner model is what load_model wants."""

    calls: list[dict] = []

    def __init__(self):
        self.model = self
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    @classmethod
    def _make(cls, kwargs):
        cls.calls.append(kwargs)
        return cls()


def _install_fake_tribev2(monkeypatch, from_pretrained):
    """Put a fake `tribev2.demo_utils.TribeModel` on the import path."""
    _FakeTribeModel.calls = []
    model_cls = type("TribeModel", (_FakeTribeModel,), {"from_pretrained": from_pretrained})
    demo_utils = types.ModuleType("tribev2.demo_utils")
    demo_utils.TribeModel = model_cls
    package = types.ModuleType("tribev2")
    package.demo_utils = demo_utils
    monkeypatch.setitem(sys.modules, "tribev2", package)
    monkeypatch.setitem(sys.modules, "tribev2.demo_utils", demo_utils)
    return model_cls


def _fake_snapshot(tmp_path: Path, revision: str = PINNED_REVISION, ckpt="best.ckpt") -> Path:
    """A directory shaped like a huggingface snapshot of `revision`."""
    snapshot = tmp_path / "hub" / "models--facebook--tribev2" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    if ckpt:
        (snapshot / ckpt).write_bytes(b"fake-weights")
    return snapshot


# ---------------------------------------------------------------------------
# ensure_offscreen_display
# ---------------------------------------------------------------------------


def _fake_pyvista(monkeypatch):
    """A pyvista with no `start_xvfb` -- the 0.46+ shape that broke gate 17."""
    module = types.ModuleType("pyvista")
    module.OFF_SCREEN = False
    monkeypatch.setitem(sys.modules, "pyvista", module)
    return module


class _FakePopen:
    """Xvfb stand-in. `returncode=None` means still running."""

    def __init__(self, returncode=None):
        self.returncode = returncode

    def poll(self):
        return self.returncode


def test_an_existing_display_is_left_alone(monkeypatch):
    monkeypatch.setenv("DISPLAY", ":0")

    def _must_not_run(*args, **kwargs):  # pragma: no cover - the point is it is not called
        raise AssertionError("should not start a framebuffer when one exists")

    _fake_pyvista(monkeypatch)
    monkeypatch.setattr(verification, "start_xvfb", _must_not_run)
    verification.ensure_offscreen_display()


def test_a_framebuffer_is_started_when_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)

    def _start(*args, **kwargs):
        import os

        os.environ["DISPLAY"] = ":99"
        return ":99"

    module = _fake_pyvista(monkeypatch)
    monkeypatch.setattr(verification, "start_xvfb", _start)
    verification.ensure_offscreen_display()
    assert module.OFF_SCREEN is True


def test_a_framebuffer_that_cannot_start_raises_with_the_fix(monkeypatch):
    """The fix belongs in the message: this fails inside a `!python` subprocess,
    where the traceback is all the user gets."""
    monkeypatch.delenv("DISPLAY", raising=False)
    _fake_pyvista(monkeypatch)
    monkeypatch.setattr(verification.shutil, "which", lambda _: None)
    with pytest.raises(VerificationError, match="xvfb"):
        verification.ensure_offscreen_display()


def test_xvfb_is_started_without_pyvistas_helper(monkeypatch, tmp_path):
    """pyvista removed `start_xvfb` in 0.46, so we launch the server ourselves.

    Guards the regression directly: the fake pyvista above has no such
    attribute, and this still has to come up.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(verification, "_XVFB_PROCESS", None)
    monkeypatch.setattr(verification.shutil, "which", lambda _: "/usr/bin/Xvfb")

    socket_dir = tmp_path / ".X11-unix"
    socket_dir.mkdir()
    (socket_dir / "X99").touch()          # the server is already serving
    monkeypatch.setattr(verification, "XVFB_SOCKET_DIR", socket_dir)

    commands = []

    def _popen(cmd, **kwargs):
        commands.append(cmd)
        return _FakePopen()

    monkeypatch.setattr(verification.subprocess, "Popen", _popen)

    assert verification.start_xvfb() == ":99"
    assert commands[0][:2] == ["Xvfb", ":99"]


def test_an_xvfb_that_dies_on_startup_raises(monkeypatch, tmp_path):
    """VTK aborts the interpreter on a missing X server -- no exception to catch.

    So an Xvfb that exits instead of serving has to be caught here, or the next
    VTK call kills the process with no traceback.
    """
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setattr(verification, "_XVFB_PROCESS", None)
    monkeypatch.setattr(verification.shutil, "which", lambda _: "/usr/bin/Xvfb")
    monkeypatch.setattr(verification, "XVFB_SOCKET_DIR", tmp_path)   # no socket ever appears
    monkeypatch.setattr(
        verification.subprocess, "Popen", lambda cmd, **kwargs: _FakePopen(returncode=1)
    )

    with pytest.raises(VerificationError, match="exited immediately"):
        verification.start_xvfb()


# ---------------------------------------------------------------------------
# hash_snapshot
# ---------------------------------------------------------------------------


def test_hash_snapshot_does_not_depend_on_where_the_hub_cache_lives(tmp_path):
    """The `.cache` skip is relative to the snapshot, not to the filesystem.

    The default hub cache is `~/.cache/huggingface`, so an absolute-path test
    excludes every file and the lock comes out empty; a Drive-backed `HF_HOME`
    has no `.cache` component and includes them all. Same revision, two locks.
    """
    under_dot_cache = tmp_path / ".cache" / "huggingface" / "snapshots" / PINNED_REVISION
    elsewhere = tmp_path / "drive" / "hf" / "snapshots" / PINNED_REVISION
    for root in (under_dot_cache, elsewhere):
        root.mkdir(parents=True)
        (root / "best.ckpt").write_bytes(b"weights")
        (root / "config.yaml").write_bytes(b"config")
        # huggingface_hub's own bookkeeping, which must stay out of the lock
        (root / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (root / ".cache" / "huggingface" / "download" / "best.ckpt.metadata").write_bytes(b"x")

    assert verification.hash_snapshot(under_dot_cache) == verification.hash_snapshot(elsewhere)
    assert sorted(verification.hash_snapshot(elsewhere)) == ["best.ckpt", "config.yaml"]


def test_hash_snapshot_raises_on_an_empty_snapshot(tmp_path):
    empty = tmp_path / "snapshots" / PINNED_REVISION
    empty.mkdir(parents=True)
    with pytest.raises(VerificationError, match="no files to checksum"):
        verification.hash_snapshot(empty)


# ---------------------------------------------------------------------------
# The published build: pin via a pre-downloaded local snapshot
# ---------------------------------------------------------------------------


def test_checkpoint_dir_build_is_given_the_pinned_snapshot(tribe_config, tmp_path, monkeypatch):
    """The real signature: no `revision`, so the pin must arrive as a path."""

    @staticmethod
    def from_pretrained(
        checkpoint_dir, checkpoint_name="best.ckpt", cache_folder=None, device="auto"
    ):
        return _FakeTribeModel._make(
            {
                "checkpoint_dir": checkpoint_dir,
                "checkpoint_name": checkpoint_name,
                "cache_folder": cache_folder,
            }
        )

    model_cls = _install_fake_tribev2(monkeypatch, from_pretrained)
    snapshot = _fake_snapshot(tmp_path)
    monkeypatch.setattr(verification, "download_checkpoint", lambda config: snapshot)

    model = verification.load_model(tribe_config, cache_folder=tmp_path / "cache")

    (call,) = model_cls.calls
    assert Path(call["checkpoint_dir"]) == snapshot
    assert PINNED_REVISION in Path(call["checkpoint_dir"]).parts
    assert call["checkpoint_name"] == "best.ckpt"
    assert model.eval_called


def test_a_build_exposing_revision_is_pinned_through_it(tribe_config, tmp_path, monkeypatch):
    """If a future build takes `revision`, no pre-download detour is needed."""

    @staticmethod
    def from_pretrained(checkpoint, revision=None, cache_folder=None):
        return _FakeTribeModel._make({"checkpoint": checkpoint, "revision": revision})

    model_cls = _install_fake_tribev2(monkeypatch, from_pretrained)

    def _must_not_download(config):  # pragma: no cover - the point is it is not called
        raise AssertionError("should not pre-download when `revision` is supported")

    monkeypatch.setattr(verification, "download_checkpoint", _must_not_download)

    verification.load_model(tribe_config)

    (call,) = model_cls.calls
    assert call == {"checkpoint": "facebook/tribev2", "revision": PINNED_REVISION}


def test_single_alternative_ckpt_is_named_explicitly(tribe_config, tmp_path, monkeypatch):
    """No `best.ckpt` but exactly one candidate: load it, do not guess silently."""

    @staticmethod
    def from_pretrained(checkpoint_dir, checkpoint_name="best.ckpt", cache_folder=None):
        return _FakeTribeModel._make({"checkpoint_name": checkpoint_name})

    model_cls = _install_fake_tribev2(monkeypatch, from_pretrained)
    snapshot = _fake_snapshot(tmp_path, ckpt="tribe_v2.ckpt")
    monkeypatch.setattr(verification, "download_checkpoint", lambda config: snapshot)

    verification.load_model(tribe_config)

    assert model_cls.calls[0]["checkpoint_name"] == "tribe_v2.ckpt"


@pytest.mark.parametrize("ckpts", [[], ["a.ckpt", "b.ckpt"]])
def test_ambiguous_checkpoint_file_raises(tribe_config, tmp_path, monkeypatch, ckpts):
    @staticmethod
    def from_pretrained(checkpoint_dir, checkpoint_name="best.ckpt"):  # pragma: no cover
        raise AssertionError("must not be reached")

    _install_fake_tribev2(monkeypatch, from_pretrained)
    snapshot = _fake_snapshot(tmp_path, ckpt=None)
    for name in ckpts:
        (snapshot / name).write_bytes(b"fake-weights")
    monkeypatch.setattr(verification, "download_checkpoint", lambda config: snapshot)

    with pytest.raises(VerificationError, match="which weights"):
        verification.load_model(tribe_config)


# ---------------------------------------------------------------------------
# The stops
# ---------------------------------------------------------------------------


def test_snapshot_without_the_revision_in_its_path_raises(tribe_config, tmp_path, monkeypatch):
    """A download that resolved elsewhere cannot be shown to be the pin."""

    @staticmethod
    def from_pretrained(checkpoint_dir, checkpoint_name="best.ckpt"):  # pragma: no cover
        raise AssertionError("must not be reached")

    _install_fake_tribev2(monkeypatch, from_pretrained)
    elsewhere = _fake_snapshot(tmp_path, revision="c" * 40)
    monkeypatch.setattr(verification, "download_checkpoint", lambda config: elsewhere)

    with pytest.raises(VerificationError, match="does not contain the pinned revision"):
        verification.load_model(tribe_config)


def test_build_with_neither_argument_raises(tribe_config, monkeypatch):
    @staticmethod
    def from_pretrained(name_or_path, device="auto"):  # pragma: no cover
        raise AssertionError("must not be reached")

    _install_fake_tribev2(monkeypatch, from_pretrained)

    with pytest.raises(VerificationError, match="neither a `revision` nor a `checkpoint_dir`"):
        verification.load_model(tribe_config)


def test_unresolved_revision_never_reaches_the_model(unresolved_config, monkeypatch):
    @staticmethod
    def from_pretrained(checkpoint_dir, checkpoint_name="best.ckpt"):  # pragma: no cover
        raise AssertionError("must not be reached")

    _install_fake_tribev2(monkeypatch, from_pretrained)

    with pytest.raises(ConfigError):
        verification.load_model(unresolved_config)


def test_fp16_still_refuses_before_any_download(tribe_config, monkeypatch):
    """The dtype stop predates the pinning work and must keep firing first."""
    from dataclasses import replace

    def _must_not_download(config):  # pragma: no cover
        raise AssertionError("should not download when the dtype is unsupported")

    monkeypatch.setattr(verification, "download_checkpoint", _must_not_download)
    _install_fake_tribev2(monkeypatch, staticmethod(lambda *a, **k: None))

    with pytest.raises(VerificationError, match="fp16"):
        verification.load_model(replace(tribe_config, precision="fp16"))
