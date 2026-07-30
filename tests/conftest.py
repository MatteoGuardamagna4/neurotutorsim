"""Shared fixtures for the Phase II and Phase III test suites.

Every test in this suite runs on CPU with no GPU and no network. Nothing here
imports torch, tribev2, nilearn or huggingface_hub, and nothing downloads
anything: the atlas is a toy fixture and the predictions are small synthetic
arrays with hand-checkable properties.

The Phase III fixtures at the bottom load the *real* `config/learners.yaml` with
a small population, rather than defining a parallel set of parameter values. A
second copy of the parameters in a fixture would drift from the config the runs
actually use, and the tests would then be checking something nobody runs.
"""

from __future__ import annotations

import warnings
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LEARNER_CONFIG_PATH = PROJECT_ROOT / "config" / "learners.yaml"


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


# ---------------------------------------------------------------------------
# Phase III: the synthetic learner engine
# ---------------------------------------------------------------------------

from src.learners.config import load_learner_config  # noqa: E402
from src.learners.curriculum import SYNTHETIC_PLACEHOLDER, load_curriculum, units_to_views  # noqa: E402
from src.learners.effort import resolve_effectiveness_columns  # noqa: E402
from src.learners.population import init_population, init_state  # noqa: E402
from src.learners.seeds import SeedStreams  # noqa: E402

#: Small enough that the whole suite stays fast, large enough that a mean over
#: learners is not dominated by sampling noise.
TEST_N_LEARNERS = 400
TEST_N_EPISODES = 12
TEST_N_UNITS = 12


def make_units_frame(n_units: int = TEST_N_UNITS, *, provenance: str = SYNTHETIC_PLACEHOLDER):
    """A synthetic units table spanning the full §5.1 difficulty range."""
    difficulties = np.resize(np.array([1, 2, 3, 4, 5]), n_units)
    return pd.DataFrame(
        {
            "unit_id": [f"SYNTH_test_{i:03d}" for i in range(n_units)],
            "domain": ["synthetic_domain"] * n_units,
            "concept": [f"synthetic_concept_{i:03d}" for i in range(n_units)],
            "difficulty": difficulties,
            "reference_answer": [f"SYNTH_ANSWER_{i:03d}" for i in range(n_units)],
            "misconception_answer": [f"SYNTH_MISCONCEPTION_{i:03d}" for i in range(n_units)],
            "misconception_description": [f"placeholder error mode {i:03d}" for i in range(n_units)],
            "provenance": [provenance] * n_units,
        }
    )


@pytest.fixture
def learner_config():
    """The real `config/learners.yaml`, with a small population and short run."""
    return load_learner_config(
        LEARNER_CONFIG_PATH, n_learners=TEST_N_LEARNERS, episodes=TEST_N_EPISODES
    )


@pytest.fixture
def units_csv(tmp_path: Path) -> Path:
    """A placeholder units table on disk."""
    path = tmp_path / "units_placeholder.csv"
    make_units_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def units_frame(units_csv: Path, learner_config):
    """Units loaded through `load_curriculum`, with `b_u` attached."""
    return load_curriculum(units_csv, learner_config.curriculum)


@pytest.fixture
def unit_views(units_frame, learner_config):
    """Immutable per-unit views, with `coverage`/`correctness` resolved."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        resolved, _ = resolve_effectiveness_columns(units_frame, learner_config.effectiveness)
    return units_to_views(resolved)


@pytest.fixture
def learner_streams(learner_config) -> SeedStreams:
    return SeedStreams(learner_config.seeds.master, learner_config.seeds.streams)


@pytest.fixture
def population_frame(learner_config, learner_streams):
    return init_population(learner_config.population.n_learners, learner_config, learner_streams)


@pytest.fixture
def learner_state(population_frame):
    return init_state(population_frame)
