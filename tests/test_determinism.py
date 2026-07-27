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
        text_hash=th, reading_rate_wpm=220.0, checkpoint_revision="a" * 40, precision="fp32"
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
        text_hash=th, reading_rate_wpm=220.0, checkpoint_revision="a" * 40, precision="fp32"
    )
    pk = parcel_cache_key(vertex_key=vk, atlas="schaefer400", parcel_weighting="area")
    assert external == [th, vk, pk]


def test_keys_are_sha256_hex_digests():
    th = text_hash(TEXT)
    vk = vertex_cache_key(
        text_hash=th, reading_rate_wpm=220.0, checkpoint_revision="a" * 40, precision="fp32"
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


def test_float_reading_rate_is_canonicalised():
    """220 and 220.0 must be the same key; an int in a config would otherwise
    silently invalidate a whole corpus."""
    th = text_hash(TEXT)
    as_int = vertex_cache_key(
        text_hash=th, reading_rate_wpm=220, checkpoint_revision="a" * 40, precision="fp32"
    )
    as_float = vertex_cache_key(
        text_hash=th, reading_rate_wpm=220.0, checkpoint_revision="a" * 40, precision="fp32"
    )
    assert as_int == as_float


@pytest.mark.parametrize(
    "kwargs",
    [
        {"reading_rate_wpm": 180.0},
        {"checkpoint_revision": "b" * 40},
        {"precision": "fp16"},
    ],
)
def test_every_key_component_actually_changes_the_key(kwargs):
    base = dict(
        text_hash=text_hash(TEXT),
        reading_rate_wpm=220.0,
        checkpoint_revision="a" * 40,
        precision="fp32",
    )
    assert vertex_cache_key(**base) != vertex_cache_key(**{**base, **kwargs})
