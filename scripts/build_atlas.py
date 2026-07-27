"""Materialize the Schaefer-400 surface parcellation as a versioned `.npz`.

    python scripts/build_atlas.py --config config/tribe.yaml

Writes `data/raw/atlas/schaefer400_fsaverage5.npz`: one parcel label per
fsaverage5 vertex, plus a parcel table (id, name, Yeo network, surface area in
mm^2). This is the only script in the project that needs `nilearn`/`nibabel`,
which is exactly why `src/tribe/aggregate.py` reads a plain `.npz` and stays
importable on a laptop.

Input: the two FreeSurfer annotation files published with the Schaefer/Yeo
2018 release, in **fsaverage5** space:

    lh.Schaefer2018_400Parcels_7Networks_order.annot
    rh.Schaefer2018_400Parcels_7Networks_order.annot

They are not fetched automatically. A parcellation in the wrong space would
mislabel cortical territory in every downstream metric while looking perfectly
healthy, so the space is asserted (10242 vertices per hemisphere) rather than
assumed, and the annot files' own checksums go into the output's version
string.

The `.npz` is not committed (`data/raw/` is gitignored), but its sha256 is
recorded in every table `aggregate.py` produces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tribe.aggregate import file_sha256, load_atlas  # noqa: E402
from src.tribe.config import load_config  # noqa: E402

FSAVERAGE5_VERTICES_PER_HEMI = 10242
BACKGROUND_LABEL = 0
DEFAULT_ANNOT = {
    "lh": "lh.Schaefer2018_400Parcels_7Networks_order.annot",
    "rh": "rh.Schaefer2018_400Parcels_7Networks_order.annot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to config/tribe.yaml")
    parser.add_argument("--lh-annot", default=None, help="left-hemisphere .annot file")
    parser.add_argument("--rh-annot", default=None, help="right-hemisphere .annot file")
    parser.add_argument("--out", default=None, help="override the output path")
    parser.add_argument("--force", action="store_true", help="overwrite an existing atlas file")
    return parser.parse_args()


def resolve_annot(explicit: str | None, hemi: str, atlas_dir: Path) -> Path:
    path = Path(explicit) if explicit else atlas_dir / DEFAULT_ANNOT[hemi]
    if not path.exists():
        raise SystemExit(
            f"{hemi} annotation not found: {path}\n\n"
            f"Phase II needs the Schaefer-400 (7 networks) FreeSurfer annotation in "
            f"fsaverage5 space. Download `{DEFAULT_ANNOT[hemi]}` from the Schaefer/Yeo 2018 "
            f"release (CBIG repository, `stable_projects/brain_parcellation/"
            f"Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/`) and "
            f"place it in {atlas_dir}, or pass --{hemi}-annot.\n\n"
            f"Do not substitute the volumetric MNI Schaefer atlas: it is a different space, "
            f"and mapping it onto a surface mesh here would silently mislabel cortex."
        )
    return path


def vertex_areas(coords: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-vertex surface area, by the standard barycentric third-of-a-triangle split.

    Summing these within a parcel gives the parcel area that equation (7) uses
    as its weight in the main specification.
    """
    triangles = np.asarray(coords, dtype=np.float64)[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    face_area = 0.5 * np.linalg.norm(cross, axis=1)
    areas = np.zeros(coords.shape[0], dtype=np.float64)
    for column in range(3):
        np.add.at(areas, faces[:, column], face_area / 3.0)
    return areas


def network_of(label_name: str) -> str:
    """Yeo network from a Schaefer label such as `7Networks_LH_Vis_1`."""
    parts = label_name.split("_")
    if len(parts) < 3:
        raise SystemExit(
            f"cannot parse a Yeo network from Schaefer label {label_name!r}; expected the "
            f"`7Networks_<hemi>_<network>_<n>` naming of the official release"
        )
    return parts[2]


def read_hemisphere(annot_path: Path, mesh_path: str, offset: int):
    """Labels (offset into a global id space), names and per-vertex areas."""
    import nibabel as nib
    from nilearn import surface

    labels, _ctab, names = nib.freesurfer.read_annot(str(annot_path))
    if labels.shape[0] != FSAVERAGE5_VERTICES_PER_HEMI:
        raise SystemExit(
            f"{annot_path} has {labels.shape[0]} vertices; fsaverage5 has "
            f"{FSAVERAGE5_VERTICES_PER_HEMI} per hemisphere. This annotation is in a "
            f"different space -- stop and get the fsaverage5 version."
        )

    coords, faces = surface.load_surf_mesh(mesh_path)
    if coords.shape[0] != FSAVERAGE5_VERTICES_PER_HEMI:
        raise SystemExit(f"fsaverage5 mesh {mesh_path} has {coords.shape[0]} vertices")

    decoded = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    # read_annot uses index 0 for the unknown/medial-wall region.
    global_labels = np.where(labels > 0, labels + offset, BACKGROUND_LABEL).astype(np.int64)
    return global_labels, decoded, vertex_areas(coords, faces)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    out_path = Path(args.out) if args.out else config.atlas_file
    atlas_dir = out_path.parent
    atlas_dir.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Rebuilding the atlas changes every parcel mapping "
            f"downstream and invalidates every cached parcel table; pass --force if that is "
            f"what you intend."
        )

    lh_annot = resolve_annot(args.lh_annot, "lh", atlas_dir)
    rh_annot = resolve_annot(args.rh_annot, "rh", atlas_dir)

    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    lh_labels, lh_names, lh_areas = read_hemisphere(lh_annot, fsaverage["pial_left"], offset=0)
    n_lh_parcels = int(len(lh_names) - 1)
    rh_labels, rh_names, rh_areas = read_hemisphere(
        rh_annot, fsaverage["pial_right"], offset=n_lh_parcels
    )

    labels = np.concatenate([lh_labels, rh_labels])
    areas_per_vertex = np.concatenate([lh_areas, rh_areas])

    parcel_names = list(lh_names[1:]) + list(rh_names[1:])
    parcel_ids = np.arange(1, len(parcel_names) + 1, dtype=np.int64)
    networks = [network_of(name) for name in parcel_names]
    parcel_areas = np.array(
        [areas_per_vertex[labels == pid].sum() for pid in parcel_ids], dtype=np.float64
    )

    empty = parcel_ids[parcel_areas <= 0]
    if empty.size:
        raise SystemExit(
            f"parcel(s) {empty.tolist()} have no vertices or zero area; the annotation and the "
            f"mesh do not agree"
        )

    version = (
        f"Schaefer2018_400Parcels_7Networks/fsaverage5 "
        f"lh:{file_sha256(lh_annot)[:12]} rh:{file_sha256(rh_annot)[:12]}"
    )
    np.savez_compressed(
        out_path,
        labels=labels,
        parcel_id=parcel_ids,
        parcel_name=np.array(parcel_names),
        network=np.array(networks),
        area_mm2=parcel_areas,
        name=np.array("schaefer400"),
        version=np.array(version),
        background_label=np.array(BACKGROUND_LABEL),
    )

    atlas = load_atlas(out_path)  # validate what we just wrote, with the real loader
    print(f"wrote {out_path}")
    print(
        f"  {atlas.n_parcels} parcels over {atlas.n_vertices} vertices "
        f"({atlas.n_background_vertices} background), "
        f"{len(set(atlas.parcels['network']))} networks"
    )
    print(f"  sha256 {atlas.file_hash}")
    print(f"  version {atlas.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
