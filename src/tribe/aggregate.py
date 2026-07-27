"""Vertex -> parcel -> network aggregation (§6.4, equations 6 and 7).

This is the step that decides the project's storage cost, and it is
irreversible: equation (6) is a mean, and a mean cannot be un-taken. Every
guard here exists because a wrong atlas mapping would silently mislabel
cortical territory in every downstream metric, contrast and RSA result.

    (6)  parcel mean:   Bbar[t, p] = (1 / |V_p|) * sum_{v in V_p} Bhat[t, v]
    (7)  network mean:  Bbar[t, n] = sum_{p in n} w_p Bbar[t, p] / sum_{p in n} w_p

with w_p the parcel's surface area (main specification) or 1 (equal weighting,
robustness).

Track B: no torch, no nilearn at import time. The atlas is read from a plain
versioned `.npz` under `data/raw/atlas/`, materialised once by
`scripts/build_atlas.py` (which is the only thing that needs nilearn).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

ALLOWED_WEIGHTING = ("area", "equal")

PARCEL_COLUMNS = (
    "stimulus_id",
    "time_index",
    "parcel_id",
    "network",
    "mean_bold",
    "sd_bold",
)

NETWORK_COLUMNS = (
    "stimulus_id",
    "time_index",
    "network",
    "mean_bold",
    "weighting",
)


class AtlasError(ValueError):
    """Raised when an atlas file is malformed or does not match the mesh."""


@dataclass(frozen=True)
class Atlas:
    """A surface parcellation of the fsaverage5 mesh.

    `labels` holds one parcel id per mesh vertex, so a vertex cannot map to two
    parcels by construction -- the array shape *is* the single-assignment
    guarantee. `background_label` marks vertices that belong to no parcel
    (medial wall); they are excluded from equation (6) and counted explicitly.
    """

    name: str
    version: str
    file_path: Path
    file_hash: str
    labels: np.ndarray
    parcels: pd.DataFrame  # parcel_id, parcel_name, network, area_mm2
    background_label: int

    @property
    def n_vertices(self) -> int:
        return int(self.labels.shape[0])

    @property
    def n_parcels(self) -> int:
        return int(len(self.parcels))

    @property
    def n_background_vertices(self) -> int:
        return int((self.labels == self.background_label).sum())

    def provenance(self) -> Dict[str, Any]:
        return {
            "atlas_name": self.name,
            "atlas_version": self.version,
            "atlas_file": str(self.file_path),
            "atlas_file_sha256": self.file_hash,
            "n_vertices": self.n_vertices,
            "n_parcels": self.n_parcels,
            "n_background_vertices": self.n_background_vertices,
        }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_atlas(path: str | Path) -> Atlas:
    """Load a versioned parcellation from `data/raw/atlas/*.npz`.

    Required arrays: `labels` (n_vertices,), `parcel_id`, `parcel_name`,
    `network`, `area_mm2` (all n_parcels,), plus scalars `name`, `version`,
    `background_label`.

    Validates that the label set and the parcel table agree exactly in both
    directions. A label with no metadata row, or a metadata row with no
    vertices, is an error -- either one means the mapping is not the atlas the
    metadata claims it is.
    """
    atlas_path = Path(path)
    if not atlas_path.exists():
        raise AtlasError(
            f"atlas file not found: {atlas_path}. Build it once with "
            f"`python scripts/build_atlas.py --config config/tribe.yaml` (needs nilearn, "
            f"Track A). It is deliberately not committed -- data/raw/ is gitignored -- but "
            f"its sha256 is recorded in every output it touches."
        )

    with np.load(atlas_path, allow_pickle=False) as handle:
        required = {"labels", "parcel_id", "parcel_name", "network", "area_mm2",
                    "name", "version", "background_label"}
        missing = sorted(required - set(handle.files))
        if missing:
            raise AtlasError(f"{atlas_path}: atlas file is missing array(s) {missing}")

        labels = np.asarray(handle["labels"]).astype(np.int64)
        parcels = pd.DataFrame(
            {
                "parcel_id": np.asarray(handle["parcel_id"]).astype(np.int64),
                "parcel_name": [str(v) for v in handle["parcel_name"]],
                "network": [str(v) for v in handle["network"]],
                "area_mm2": np.asarray(handle["area_mm2"]).astype(float),
            }
        )
        name = str(handle["name"])
        version = str(handle["version"])
        background_label = int(handle["background_label"])

    if labels.ndim != 1:
        raise AtlasError(f"{atlas_path}: labels must be 1-D (one per vertex); got {labels.shape}")
    if parcels["parcel_id"].duplicated().any():
        dupes = sorted(parcels.loc[parcels["parcel_id"].duplicated(), "parcel_id"])
        raise AtlasError(f"{atlas_path}: duplicate parcel_id(s) in the parcel table: {dupes}")
    if (parcels["area_mm2"] <= 0).any():
        raise AtlasError(f"{atlas_path}: every parcel needs a positive surface area")

    observed = set(np.unique(labels).tolist()) - {background_label}
    declared = set(parcels["parcel_id"].tolist())
    if observed - declared:
        raise AtlasError(
            f"{atlas_path}: vertex labels {sorted(observed - declared)} have no row in the "
            f"parcel table"
        )
    if declared - observed:
        raise AtlasError(
            f"{atlas_path}: parcel(s) {sorted(declared - observed)} are declared but no "
            f"vertex maps to them"
        )

    return Atlas(
        name=name,
        version=version,
        file_path=atlas_path,
        file_hash=file_sha256(atlas_path),
        labels=labels,
        parcels=parcels.sort_values("parcel_id").reset_index(drop=True),
        background_label=background_label,
    )


def assert_covers_mesh(atlas: Atlas, n_vertices: int) -> None:
    """The mapping must cover exactly the mesh, vertex for vertex.

    `len(labels) == n_vertices` is both the coverage check and the
    no-double-assignment check: one label slot per vertex, no more, no less.
    """
    if atlas.n_vertices != n_vertices:
        raise AtlasError(
            f"atlas '{atlas.name}' maps {atlas.n_vertices} vertices but the prediction "
            f"array has {n_vertices}. The vertex->parcel mapping must cover exactly the "
            f"mesh vertex count; a partial mapping would silently drop cortex."
        )


def vertices_to_parcels(
    B_hat: np.ndarray,
    atlas: Atlas,
    weighting: str = "equal",
    *,
    stimulus_id: str,
) -> pd.DataFrame:
    """Equation (6): unweighted mean over the vertex set of each parcel.

    Returns long format: stimulus_id, time_index, parcel_id, network,
    mean_bold, sd_bold. `sd_bold` is the spread *across vertices within the
    parcel* at that timepoint -- it is the information equation (6) throws
    away, kept as a scalar so aggregation loss is at least visible.

    `weighting` does not change this computation: equation (6) is defined as a
    plain unweighted mean, and the weighting choice applies at the network
    level (equation 7). It is accepted here and carried into the output
    provenance so a parcel table records which scheme its downstream network
    aggregation used, rather than that choice living only in a config file.
    """
    if weighting not in ALLOWED_WEIGHTING:
        raise ValueError(f"weighting must be one of {list(ALLOWED_WEIGHTING)}; got {weighting!r}")

    array = np.asarray(B_hat, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(
            f"B_hat must be a 2-D (n_timesteps, n_vertices) array; got shape {array.shape}"
        )
    n_timesteps, n_vertices = array.shape
    assert_covers_mesh(atlas, n_vertices)

    frames = []
    for _, parcel in atlas.parcels.iterrows():
        parcel_id = int(parcel["parcel_id"])
        member_vertices = atlas.labels == parcel_id
        block = array[:, member_vertices]
        if block.shape[1] == 0:  # unreachable: load_atlas rejects empty parcels
            raise AtlasError(f"parcel {parcel_id} has no vertices")
        frames.append(
            pd.DataFrame(
                {
                    "stimulus_id": stimulus_id,
                    "time_index": np.arange(n_timesteps, dtype=np.int64),
                    "parcel_id": parcel_id,
                    "network": str(parcel["network"]),
                    "mean_bold": block.mean(axis=1),
                    # ddof=0: this is the spread of the vertices we have, not an
                    # estimate of a population from a sample of them.
                    "sd_bold": block.std(axis=1, ddof=0),
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)[list(PARCEL_COLUMNS)]
    out.attrs["provenance"] = {
        **atlas.provenance(),
        "equation": "6",
        "vertex_to_parcel": "unweighted mean",
        "parcel_weighting": weighting,
        "n_timesteps": int(n_timesteps),
    }
    return out


def parcels_to_networks(parcel_df: pd.DataFrame, weighting: str, atlas: Atlas) -> pd.DataFrame:
    """Equation (7): area-weighted (main) or equally weighted (robustness) mean
    of parcels within each network.

    The chosen weighting is written into a `weighting` column of the output, not
    left implicit in a config file, so a table can always answer how it was
    built without its config.
    """
    if weighting not in ALLOWED_WEIGHTING:
        raise ValueError(f"weighting must be one of {list(ALLOWED_WEIGHTING)}; got {weighting!r}")
    missing = [c for c in ("stimulus_id", "time_index", "parcel_id", "network", "mean_bold")
               if c not in parcel_df.columns]
    if missing:
        raise ValueError(f"parcel table is missing column(s) {missing}")

    areas = atlas.parcels.set_index("parcel_id")["area_mm2"]
    unknown = sorted(set(parcel_df["parcel_id"]) - set(areas.index))
    if unknown:
        raise AtlasError(
            f"parcel table contains parcel_id(s) {unknown} that atlas '{atlas.name}' does "
            f"not define; refusing to weight an unknown parcel"
        )

    frame = parcel_df.copy()
    frame["_w"] = 1.0 if weighting == "equal" else frame["parcel_id"].map(areas).astype(float)
    frame["_wx"] = frame["_w"] * frame["mean_bold"]

    grouped = frame.groupby(["stimulus_id", "time_index", "network"], as_index=False, sort=True)[
        ["_w", "_wx"]
    ].sum()
    grouped["mean_bold"] = grouped["_wx"] / grouped["_w"]
    grouped["weighting"] = weighting

    out = grouped[list(NETWORK_COLUMNS)]
    out.attrs["provenance"] = {**atlas.provenance(), "equation": "7", "weighting": weighting}
    return out


def aggregate_prediction(
    B_hat: np.ndarray,
    atlas: Atlas,
    *,
    stimulus_id: str,
    weighting: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Convenience wrapper: equation (6), with atlas provenance merged into
    `.attrs` alongside the prediction metadata. Used by the inference loop.
    """
    parcels = vertices_to_parcels(B_hat, atlas, weighting, stimulus_id=stimulus_id)
    if metadata:
        parcels.attrs["provenance"] = {**parcels.attrs.get("provenance", {}), **metadata}
    return parcels
