"""Named RNG streams for Phase III (§10.1 bitwise reproducibility).

One master seed lives in `config/learners.yaml`. Every random draw in the
learner engine comes from a named substream spawned off it, never from the
global `np.random` functions -- those share one process-wide state, so a single
stray call anywhere reorders every draw downstream and silently breaks the
determinism guarantee.

Four streams (§3.1 of the Phase III task spec):

===============  =========================================================
`population`     the eq. (15) copula draw and the eq. (16) per-learner
                 parameters -- consumed once, at initialisation
`responses`      correctness, confidence noise and support requests
`tutor`          the tutor's error-diagnosis success
`checkpoints`    the §7.7 assessment probes
===============  =========================================================

Separating them means a change to, say, the checkpoint schedule cannot shift
the response draws of the episodes before it.

Labels and independence
-----------------------
`derive(label)` produces an independent family of the same four streams, keyed
by a *string* label hashed with `hashlib`. Conditions are labelled this way, so
running `[traditional, ai_scaffolding]` gives the traditional arm exactly the
draws it would have received from `[traditional, ai_scaffolding,
ai_substitution]`. `hash()` would not do: it is salted per process, so the
label key -- and every draw under it -- would differ between two runs that a
determinism check compares.

Track B. numpy only; never torch.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

from src.learners.config import REQUIRED_STREAMS

#: Bytes of the sha256 digest folded into a `SeedSequence.spawn_key` entry.
#: Eight bytes is 64 bits -- far beyond any plausible number of labels, and
#: `SeedSequence` accepts arbitrary non-negative integers.
_LABEL_KEY_BYTES = 8


def label_key(label: str) -> int:
    """Stable non-negative integer for a string label.

    Uses `hashlib.sha256`, not `hash()`: the built-in is salted per process, so
    a spawn key derived from it would differ between runs and silently break
    reproducibility.
    """
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:_LABEL_KEY_BYTES], "big")


class SeedStreams:
    """The named substreams of one master seed.

    `generator(name)` memoises, so repeated calls return the *same*
    `np.random.Generator` and draws advance monotonically through the stream.
    Getting a fresh generator each call would silently repeat the same numbers
    every episode.

    Parameters
    ----------
    master
        The master seed from `seeds.master`.
    names
        Stream names, in the order the config declares them. The order fixes
        which spawned child each name receives.
    labels
        Optional chain of string labels distinguishing this family from others
        derived off the same master seed.
    """

    __slots__ = ("master", "names", "labels", "_sequences", "_generators")

    def __init__(
        self,
        master: int,
        names: Sequence[str] = REQUIRED_STREAMS,
        *,
        labels: Iterable[str] = (),
    ) -> None:
        self.master = int(master)
        self.names: Tuple[str, ...] = tuple(names)
        self.labels: Tuple[str, ...] = tuple(labels)
        if not self.names:
            raise ValueError("SeedStreams needs at least one stream name")
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"stream names must be unique; got {list(self.names)}")
        base = np.random.SeedSequence(
            entropy=self.master,
            spawn_key=tuple(label_key(label) for label in self.labels),
        )
        self._sequences: Dict[str, np.random.SeedSequence] = dict(
            zip(self.names, base.spawn(len(self.names)))
        )
        self._generators: Dict[str, np.random.Generator] = {}

    def sequence(self, name: str) -> np.random.SeedSequence:
        """The `SeedSequence` behind one named stream."""
        if name not in self._sequences:
            raise KeyError(
                f"unknown seed stream {name!r}; this family carries {list(self.names)}"
            )
        return self._sequences[name]

    def generator(self, name: str) -> np.random.Generator:
        """The memoised `Generator` for one named stream."""
        if name not in self._generators:
            self._generators[name] = np.random.default_rng(self.sequence(name))
        return self._generators[name]

    def derive(self, label: str) -> "SeedStreams":
        """An independent family of the same streams, keyed by `label`."""
        return SeedStreams(self.master, self.names, labels=(*self.labels, label))

    def as_dict(self) -> Dict[str, object]:
        """Provenance record for the run log."""
        return {
            "master": self.master,
            "streams": list(self.names),
            "labels": list(self.labels),
        }

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return f"SeedStreams(master={self.master}, labels={self.labels!r})"
