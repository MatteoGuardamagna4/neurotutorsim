"""Content-addressed cache for TRIBE predictions (§4.3 idempotence, §6.3 storage).

The one fact that shapes this whole module:

    **Parcel averaging is one-way.** Once a (T, ~20484) vertex array has been
    reduced to (T, 400) parcel means, the vertices are gone. No later analysis
    can recover them, and re-deriving them costs another GPU run against a
    gated model.

So the cache is *hybrid*, not uniform:

* parcel level -- every stimulus in the full grid, kept permanently (~1.2 GB);
* vertex level -- only the ids in `config.vertex_retention_set` (~3.5 GB).

and `resolve()` is deliberately three-branched: a parcel entry is a hit for a
parcel request and a *miss* for a vertex request. It never fabricates vertex
data from parcel means.

Keys are nested and content-addressed:

    vertex_key = sha256(text_hash, checkpoint_revision, precision)
    parcel_key = sha256(vertex_key, atlas, parcel_weighting)

Hashing uses `hashlib` over canonical JSON -- never Python's `hash()`, which is
salted per process and would make the cache non-reproducible across sessions
(§10.1).

Track A/B: no torch import.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from src.tribe.config import TribeConfig

MANIFEST_NAME = "manifest.parquet"

#: Every manifest row carries these columns, in this order.
MANIFEST_COLUMNS = (
    "vertex_key",
    "parcel_key",
    "stimulus_id",
    "level",
    "path",
    "size_bytes",
    "timestamp_utc",
    "text_hash",
    "reading_rate_wpm",
    "checkpoint",
    "revision",
    "precision",
    "atlas",
    "parcel_weighting",
    "n_timesteps",
    "n_vertices",
    "n_parcels",
    "runtime_s",
)

MISS_NO_ENTRY = "no_entry"
MISS_PARCEL_ONLY_VERTEX_REQUIRED = "parcel_only_vertex_required"


class CacheError(RuntimeError):
    """Raised when the cache is structurally inconsistent."""


def _canonical(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    """sha256 of the stimulus body, as stored in metadata and cache keys.

    Newlines are normalised so a CRLF checkout and an LF checkout of the same
    stimulus produce the same key. Nothing else is normalised: whitespace and
    casing are part of the stimulus.
    """
    normalised = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


def vertex_cache_key(
    *, text_hash: str, checkpoint_revision: str, precision: str) -> str:
    """Identity of a raw TRIBE prediction (§4.5).

    Anything that changes the numbers changes the key: the stimulus text, the
    pinned checkpoint revision, the numeric precision.

    `reading_rate_wpm` is deliberately **not** here. It was, while §6.2's onsets
    were being written into TRIBE's events frame. They no longer are -- TRIBE
    gets its own frame and derives timings from the TTS waveform it renders (see
    `src/tribe/events.py`), so r cannot change a single predicted value. Keeping
    it in the key would invalidate correct cache entries on a robustness sweep
    over r and buy nothing: every recomputation would return the same array at
    the cost of another GPU run against a gated model. r stays in the manifest,
    where it records the configuration a run was made under.
    """
    return _sha256(
        {
            "text_hash": text_hash,
            "checkpoint_revision": checkpoint_revision,
            "precision": precision,
        }
    )


def parcel_cache_key(*, vertex_key: str, atlas: str, parcel_weighting: str) -> str:
    """Identity of a parcel aggregation, nested under its vertex prediction."""
    return _sha256(
        {"vertex_key": vertex_key, "atlas": atlas, "parcel_weighting": parcel_weighting}
    )


def keys_for(text: str, config: TribeConfig) -> tuple[str, str]:
    """Both cache keys for `text` under `config`."""
    th = text_hash(text)
    vk = vertex_cache_key(
        text_hash=th,
        checkpoint_revision=config.checkpoint_revision,
        precision=config.precision,
    )
    pk = parcel_cache_key(
        vertex_key=vk, atlas=config.atlas, parcel_weighting=config.parcel_weighting
    )
    return vk, pk


@dataclass(frozen=True)
class CacheResult:
    """Outcome of a `resolve()` call.

    `status` is one of "hit_vertex", "hit_parcel", "miss". On a miss, `reason`
    says which of the two miss kinds it is -- the caller must be able to tell
    "never computed" from "computed, but the vertices were deliberately
    discarded by the retention policy".
    """

    status: str
    stimulus_id: str
    vertex_key: str
    parcel_key: str
    vertex_path: Path
    parcel_path: Path
    reason: Optional[str] = None
    data: Optional[Any] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def is_hit(self) -> bool:
        return self.status != "miss"

    @property
    def needs_inference(self) -> bool:
        return self.status == "miss"


@dataclass
class TribeCache:
    """Filesystem cache rooted at `config.cache_root`."""

    config: TribeConfig
    root: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root = Path(self.config.cache_root)

    # -- paths -----------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    def vertex_path(self, vertex_key: str) -> Path:
        return self.root / "vertex" / vertex_key[:2] / f"{vertex_key}.npz"

    def parcel_path(self, parcel_key: str) -> Path:
        return self.root / "parcel" / parcel_key[:2] / f"{parcel_key}.parquet"

    # -- resolution ------------------------------------------------------

    def resolve(
        self,
        stimulus_id: str,
        text: str,
        *,
        need_vertex: bool,
        load_data: bool = True,
    ) -> CacheResult:
        """Three-branch cache lookup (§4.5).

        1. parcel entry exists and `need_vertex` is False -> parcel hit, no GPU work;
        2. vertex entry exists -> vertex hit (parcels can be derived from it);
        3. neither -> miss, inference required.

        With `need_vertex=True`, a parcel-only entry is **not** a hit: it comes
        back as a miss with reason `parcel_only_vertex_required`. Vertex data is
        never reconstructed from parcel means.

        `load_data=False` performs the lookup without reading the payload --
        what the inference loop wants when all it needs is skip/don't-skip.
        """
        vk, pk = keys_for(text, self.config)
        vpath, ppath = self.vertex_path(vk), self.parcel_path(pk)
        vertex_exists, parcel_exists = vpath.exists(), ppath.exists()

        if vertex_exists:
            data, meta = (self.read_vertex(vk) if load_data else (None, None))
            return CacheResult(
                status="hit_vertex",
                stimulus_id=stimulus_id,
                vertex_key=vk,
                parcel_key=pk,
                vertex_path=vpath,
                parcel_path=ppath,
                data=data,
                metadata=meta,
            )

        if parcel_exists:
            if need_vertex:
                return CacheResult(
                    status="miss",
                    stimulus_id=stimulus_id,
                    vertex_key=vk,
                    parcel_key=pk,
                    vertex_path=vpath,
                    parcel_path=ppath,
                    reason=MISS_PARCEL_ONLY_VERTEX_REQUIRED,
                )
            data = self.read_parcel(pk) if load_data else None
            return CacheResult(
                status="hit_parcel",
                stimulus_id=stimulus_id,
                vertex_key=vk,
                parcel_key=pk,
                vertex_path=vpath,
                parcel_path=ppath,
                data=data,
            )

        return CacheResult(
            status="miss",
            stimulus_id=stimulus_id,
            vertex_key=vk,
            parcel_key=pk,
            vertex_path=vpath,
            parcel_path=ppath,
            reason=MISS_NO_ENTRY,
        )

    # -- writes ----------------------------------------------------------

    def write_vertex(
        self, vertex_key: str, predictions: np.ndarray, metadata: Mapping[str, Any]
    ) -> Path:
        """Store the complete time-resolved `(T, V)` array as compressed npz.

        §6.3 forbids storing a static per-vertex summary in place of the full
        matrix, so a 1-D input is rejected rather than reshaped.
        """
        array = np.asarray(predictions)
        if array.ndim != 2:
            raise CacheError(
                f"vertex predictions must be a 2-D (n_timesteps, n_vertices) array; got "
                f"shape {array.shape}. §6.3 requires the complete time-resolved matrix -- "
                f"a per-vertex summary is not an acceptable substitute."
            )
        path = self.vertex_path(vertex_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            predictions=array.astype(np.float32, copy=False),
            metadata_json=np.array(_canonical(dict(metadata))),
        )
        self.append_manifest(
            level="vertex", path=path, metadata=metadata, vertex_key=vertex_key
        )
        return path

    def write_parcel(
        self, parcel_key: str, parcel_df: pd.DataFrame, metadata: Mapping[str, Any]
    ) -> Path:
        """Store the long-format parcel table as parquet."""
        if parcel_df.empty:
            raise CacheError("refusing to cache an empty parcel table")
        path = self.parcel_path(parcel_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = parcel_df.copy()
        frame.attrs = {}
        frame.to_parquet(path, index=False, compression="snappy")
        self.append_manifest(
            level="parcel", path=path, metadata=metadata, parcel_key=parcel_key
        )
        return path

    # -- reads -----------------------------------------------------------

    def read_vertex(self, vertex_key: str) -> tuple[np.ndarray, Dict[str, Any]]:
        path = self.vertex_path(vertex_key)
        if not path.exists():
            raise CacheError(f"no vertex entry for key {vertex_key} at {path}")
        with np.load(path, allow_pickle=False) as handle:
            array = handle["predictions"]
            meta = json.loads(str(handle["metadata_json"]))
        return array, meta

    def read_parcel(self, parcel_key: str) -> pd.DataFrame:
        path = self.parcel_path(parcel_key)
        if not path.exists():
            raise CacheError(f"no parcel entry for key {parcel_key} at {path}")
        return pd.read_parquet(path)

    # -- manifest --------------------------------------------------------

    def manifest(self) -> pd.DataFrame:
        """The append-only manifest, one row per cache entry."""
        if not self.manifest_path.exists():
            return pd.DataFrame(columns=list(MANIFEST_COLUMNS))
        return pd.read_parquet(self.manifest_path)

    def append_manifest(
        self,
        *,
        level: str,
        path: Path,
        metadata: Mapping[str, Any],
        vertex_key: Optional[str] = None,
        parcel_key: Optional[str] = None,
    ) -> None:
        """Append one row. Written per item, never batched at the end.

        Colab sessions die at 12h, on disconnect, or on idle; a manifest
        assembled in memory and flushed once is a manifest that is usually lost.
        """
        row: Dict[str, Any] = {column: None for column in MANIFEST_COLUMNS}
        row.update({k: v for k, v in metadata.items() if k in MANIFEST_COLUMNS})
        row["vertex_key"] = vertex_key or metadata.get("vertex_key")
        row["parcel_key"] = parcel_key or metadata.get("parcel_key")
        row["level"] = level
        row["path"] = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        row["size_bytes"] = path.stat().st_size if path.exists() else None
        row["timestamp_utc"] = metadata.get(
            "timestamp_utc", datetime.now(timezone.utc).isoformat()
        )
        if row["stimulus_id"] is None:
            raise CacheError(
                "manifest row requires a stimulus_id in the metadata; refusing to write "
                "an unattributable cache entry"
            )

        new_row = pd.DataFrame([row], columns=list(MANIFEST_COLUMNS))
        existing = self.manifest()
        # Concatenating onto an all-NA empty frame changes dtypes in pandas 2.x;
        # the first row simply *is* the manifest.
        frame = new_row if existing.empty else pd.concat([existing, new_row], ignore_index=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self.manifest_path, index=False)

    def verify_manifest(self) -> Dict[str, List[str]]:
        """Cross-check manifest rows against files on disk.

        Reports `missing` (in the manifest, absent on disk) and `orphans` (on
        disk, absent from the manifest). Deletes nothing: a cache entry costs a
        GPU run against a gated model, so removal is always a human decision.
        """
        manifest = self.manifest()
        recorded = {str(p) for p in manifest["path"].dropna()} if not manifest.empty else set()

        missing = sorted(p for p in recorded if not (self.root / p).exists())

        on_disk = set()
        for pattern in ("vertex/**/*.npz", "parcel/**/*.parquet"):
            for found in self.root.glob(pattern):
                on_disk.add(str(found.relative_to(self.root)))
        orphans = sorted(on_disk - recorded)

        return {"missing": missing, "orphans": orphans}
