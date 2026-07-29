"""Cache keys must be stable across processes (§10.1).

Python's built-in `hash()` is salted per interpreter process (PYTHONHASHSEED),
so a cache key derived from it would silently miss every entry written by an
earlier session. Colab sessions are short and disposable; this is not a
hypothetical failure mode.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.tribe.cache import parcel_cache_key, text_hash, vertex_cache_key

REPO_ROOT = Path(__file__).resolve().parent.parent

TEXT = "Break-even analysis answers a simple question."


def _run_in_fresh_process(snippet: str, hashseed: str) -> str:
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(snippet)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": hashseed},
    )
    if result.returncode != 0:
        raise AssertionError(f"subprocess failed:\n{result.stderr}")
    return result.stdout.strip()


SNIPPET = """
    import sys
    sys.path.insert(0, ".")
    from src.tribe.cache import text_hash, vertex_cache_key, parcel_cache_key
    th = text_hash("Break-even analysis answers a simple question.")
    vk = vertex_cache_key(
        text_hash=th, checkpoint_revision="a" * 40, precision="fp32"
    )
    pk = parcel_cache_key(vertex_key=vk, atlas="schaefer400", parcel_weighting="area")
    print(th, vk, pk)
"""


def test_keys_are_identical_across_processes_with_different_hash_seeds():
    first = _run_in_fresh_process(SNIPPET, hashseed="0")
    second = _run_in_fresh_process(SNIPPET, hashseed="12345")
    assert first == second
    assert len(first.split()) == 3


def test_keys_match_the_value_computed_in_this_process():
    external = _run_in_fresh_process(SNIPPET, hashseed="99").split()
    th = text_hash(TEXT)
    vk = vertex_cache_key(
        text_hash=th, checkpoint_revision="a" * 40, precision="fp32"
    )
    pk = parcel_cache_key(vertex_key=vk, atlas="schaefer400", parcel_weighting="area")
    assert external == [th, vk, pk]


def test_keys_are_sha256_hex_digests():
    th = text_hash(TEXT)
    vk = vertex_cache_key(
        text_hash=th, checkpoint_revision="a" * 40, precision="fp32"
    )
    for key in (th, vk):
        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)


def test_cache_module_does_not_use_builtin_hash():
    """A grep-level guard: `hash(` on a string is the failure this file exists
    to prevent, and it is easy to reintroduce."""
    source = (REPO_ROOT / "src" / "tribe" / "cache.py").read_text(encoding="utf-8")
    offending = [
        line.strip()
        for line in source.splitlines()
        if "hash(" in line
        and "_hash(" not in line
        and "hashlib" not in line
        and "cache_key" not in line
        and not line.strip().startswith(("#", "*", '"'))
    ]
    assert offending == [], f"cache.py must not call the builtin hash(): {offending}"


def test_reading_rate_is_not_part_of_the_vertex_key():
    """r does not reach TRIBE, so it must not gate the cache.

    TRIBE renders its own TTS audio and derives every timing from it; the §6.2
    rate governs Track B. Were r still in the key, a robustness sweep over
    {180, 220, 260} would recompute three identical arrays at the cost of three
    GPU runs against a gated model."""
    import inspect

    assert "reading_rate_wpm" not in inspect.signature(vertex_cache_key).parameters


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text_hash": "c" * 64},
        {"checkpoint_revision": "b" * 40},
        {"precision": "fp16"},
    ],
)
def test_every_key_component_actually_changes_the_key(kwargs):
    base = dict(
        text_hash=text_hash(TEXT),
        checkpoint_revision="a" * 40,
        precision="fp32",
    )
    assert vertex_cache_key(**base) != vertex_cache_key(**{**base, **kwargs})


# -- cuBLAS workspace configuration ------------------------------------------
#
# On CUDA >= 10.2 cuBLAS chooses a workspace per stream, so GEMM results depend
# on stream scheduling. torch refuses to run one under
# `use_deterministic_algorithms(True)` unless CUBLAS_WORKSPACE_CONFIG is set --
# and TRIBE hits exactly that path inside Llama's rotary embedding.


def _require_cublas():
    pytest.importorskip("torch", reason="Track A only; a laptop sync installs no torch")
    from src.tribe.inference import _require_deterministic_cublas

    return _require_deterministic_cublas


def test_cublas_workspace_is_configured_when_unset(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    _require_cublas()()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


@pytest.mark.parametrize("value", [":4096:8", ":16:8"])
def test_an_already_reproducible_setting_is_left_alone(monkeypatch, value):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", value)
    _require_cublas()()
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == value


def test_a_non_reproducible_setting_raises_rather_than_being_overwritten(monkeypatch):
    """Someone set this deliberately. Silently replacing it would substitute our
    judgement for theirs on a variable that changes numerical results."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":2:2")
    with pytest.raises(RuntimeError, match="not one of the reproducible settings"):
        _require_cublas()()
