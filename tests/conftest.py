"""Shared fixtures for the Phase II test suite.

Every test in this suite runs on CPU with no GPU and no network. Nothing here
imports torch, tribev2, nilearn or huggingface_hub, and nothing downloads
anything: the atlas is a toy fixture and the predictions are small synthetic
arrays with hand-checkable properties.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tribe.aggregate import Atlas
from src.tribe.config import (
    AnalysisConfig,
    MetricsConfig,
    TribeConfig,
    UNRESOLVED_REVISION,
)

PINNED_REVISION = "a" * 40


@pytest.fixture
def toy_atlas(tmp_path: Path) -> Atlas:
    """A 6-vertex mesh in 3 parcels across 2 networks.

    Parcel 1: vertices 0,1   (network A, area 2.0)
    Parcel 2: vertices 2,3   (network A, area 4.0)
    Parcel 3: vertices 4,5   (network B, area 6.0)

    Small enough that every aggregation result can be computed by hand.
    """
    atlas_path = tmp_path / "toy_atlas.npz"
    atlas_path.write_bytes(b"toy-atlas-fixture")  # hashed, never parsed
    return Atlas(
        name="toy3",
        version="test-1",
        file_path=atlas_path,
        file_hash="0" * 64,
        labels=np.array([1, 1, 2, 2, 3, 3], dtype=np.int64),
        parcels=pd.DataFrame(
            {
                "parcel_id": [1, 2, 3],
                "parcel_name": ["p1", "p2", "p3"],
                "network": ["A", "A", "B"],
                "area_mm2": [2.0, 4.0, 6.0],
            }
        ),
        background_label=0,
    )


@pytest.fixture
def metrics_config() -> MetricsConfig:
    return MetricsConfig(
        tr_seconds=1.0,
        instructional_window_s=None,
        smoothing_kernel="boxcar",
        smoothing_width_s=1.0,  # width 1 at TR 1 => no smoothing, keeps tests exact
        baseline_percentile=10.0,
        sustained_threshold_sd=1.0,
    )


@pytest.fixture
def analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        n_bootstrap=50,
        n_permutations=50,
        winsor_lower_pct=1.0,
        winsor_upper_pct=99.0,
        corpus_n_units=10,
    )


@pytest.fixture
def tribe_config(tmp_path: Path, metrics_config, analysis_config) -> TribeConfig:
    """A fully-formed config pointing at a temporary cache root."""
    retention = tmp_path / "retention.txt"
    retention.write_text("be_001_traditional_primary\n", encoding="utf-8")
    return TribeConfig(
        checkpoint="facebook/tribev2",
        checkpoint_revision=PINNED_REVISION,
        precision="fp32",
        reading_rate_wpm=220.0,
        atlas="schaefer400",
        parcel_weighting="area",
        batch_size=1,
        vertex_retention_set=retention,
        cache_root=tmp_path / "cache",
        atlas_file=tmp_path / "atlas.npz",
        checkpoints_lock=tmp_path / "checkpoints.lock",
        master_seed=7,
        metrics=metrics_config,
        analysis=analysis_config,
        project_root=tmp_path,
        source_path=tmp_path / "tribe.yaml",
    )


@pytest.fixture
def unresolved_config(tribe_config) -> TribeConfig:
    from dataclasses import replace

    return replace(tribe_config, checkpoint_revision=UNRESOLVED_REVISION)


def make_parcel_frame(
    stimulus_id: str, matrix: np.ndarray, networks: dict[int, str]
) -> pd.DataFrame:
    """Long-format parcel table from an (n_parcels, n_timesteps) matrix."""
    rows = []
    for row, (parcel_id, network) in enumerate(sorted(networks.items())):
        for t, value in enumerate(matrix[row]):
            rows.append(
                {
                    "stimulus_id": stimulus_id,
                    "time_index": t,
                    "parcel_id": parcel_id,
                    "network": network,
                    "mean_bold": float(value),
                    "sd_bold": 0.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_metrics() -> pd.DataFrame:
    """A metrics table for 12 units x 3 conditions x 4 parcels, metric `auc`.

    Twelve units clears decision gate 16, so guard tests can distinguish
    "raises because the sample is too small" from "raises for another reason".
    """
    rng = np.random.default_rng(0)
    rows = []
    for unit in range(12):
        base = rng.normal(0, 1, size=4)
        for condition, shift in (
            ("traditional", 0.0),
            ("ai_scaffolding", 0.5),
            ("ai_substitution", -0.5),
        ):
            values = base + shift + rng.normal(0, 0.1, size=4)
            for parcel_id, value in enumerate(values, start=1):
                rows.append(
                    {
                        "unit_id": f"be_{unit:03d}",
                        "condition": condition,
                        "stimulus_id": f"be_{unit:03d}_{condition}_primary",
                        "parcel_id": parcel_id,
                        "network": "A" if parcel_id < 3 else "B",
                        "metric": "auc",
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)
