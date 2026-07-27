"""Frozen configuration for Phase II (§6.1).

One YAML file (`config/tribe.yaml`) drives Track A (GPU inference) and Track B
(CPU analysis). The dataclasses here are frozen and validate on construction:
an unknown key, a malformed revision SHA, or an out-of-set precision raises
rather than being coerced to something plausible. That strictness is the point
-- a cache key is derived from these fields (§4.5), so a silently accepted
value would produce silently wrong cache hits.

This module imports nothing from torch and is safe on a laptop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

#: Sentinel written into `config/tribe.yaml` before anyone has been able to
#: resolve the real HuggingFace revision of `facebook/tribev2` (which needs an
#: approved Llama-3.2-3B gate + a network round trip). It is a *valid* 40-hex
#: string so that config loading works for Track B, but Track A refuses to use
#: it -- see `require_resolved_revision`.
UNRESOLVED_REVISION = "0" * 40

ALLOWED_PRECISION = ("fp32", "fp16")
ALLOWED_ATLAS = ("schaefer400",)
ALLOWED_WEIGHTING = ("area", "equal")
ALLOWED_SMOOTHING = ("boxcar", "gaussian")

#: Robustness reading rates permitted by §6.2 alongside the 220 wpm main
#: specification. Anything else is a typo, not a scenario.
ALLOWED_READING_RATES = (180.0, 220.0, 260.0)


class ConfigError(ValueError):
    """Raised for any malformed or unknown configuration entry."""


def _reject_unknown(section: str, data: Mapping[str, Any], allowed: Sequence[str]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(
            f"unknown key(s) in config section {section!r}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}. Configuration is not coerced -- "
            f"add the field to the dataclass or fix the spelling."
        )


def _require(section: str, data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise ConfigError(f"config section {section!r} is missing required key {key!r}")
    return data[key]


@dataclass(frozen=True)
class MetricsConfig:
    """Parameters for §6.5 neural metrics.

    Not enumerated in the Phase II task spec §4.1 field list, but §4.7 requires
    the smoothing kernel and width to come from config and be recorded in the
    output, and the sustained-engagement threshold and instructional window are
    equally free parameters that must not be hard-coded.
    """

    tr_seconds: float
    instructional_window_s: Optional[Tuple[float, float]]
    smoothing_kernel: str
    smoothing_width_s: float
    baseline_percentile: float
    sustained_threshold_sd: float

    def __post_init__(self) -> None:
        if self.tr_seconds <= 0:
            raise ConfigError(f"metrics.tr_seconds must be > 0; got {self.tr_seconds}")
        if self.smoothing_kernel not in ALLOWED_SMOOTHING:
            raise ConfigError(
                f"metrics.smoothing_kernel must be one of {list(ALLOWED_SMOOTHING)}; "
                f"got {self.smoothing_kernel!r}"
            )
        if self.smoothing_width_s <= 0:
            raise ConfigError(f"metrics.smoothing_width_s must be > 0; got {self.smoothing_width_s}")
        if not 0.0 <= self.baseline_percentile <= 100.0:
            raise ConfigError(
                f"metrics.baseline_percentile must be in [0, 100]; got {self.baseline_percentile}"
            )
        if self.sustained_threshold_sd < 0:
            raise ConfigError(
                f"metrics.sustained_threshold_sd must be >= 0; got {self.sustained_threshold_sd}"
            )
        if self.instructional_window_s is not None:
            start, end = self.instructional_window_s
            if not end > start >= 0:
                raise ConfigError(
                    f"metrics.instructional_window_s must be [start, end] with "
                    f"0 <= start < end; got {self.instructional_window_s}"
                )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsConfig":
        allowed = [f.name for f in fields(cls)]
        _reject_unknown("metrics", data, allowed)
        window = data.get("instructional_window_s")
        if window is not None:
            if not isinstance(window, (list, tuple)) or len(window) != 2:
                raise ConfigError(
                    "metrics.instructional_window_s must be null or a two-element "
                    f"[start, end] list; got {window!r}"
                )
            window = (float(window[0]), float(window[1]))
        return cls(
            tr_seconds=float(_require("metrics", data, "tr_seconds")),
            instructional_window_s=window,
            smoothing_kernel=str(_require("metrics", data, "smoothing_kernel")),
            smoothing_width_s=float(_require("metrics", data, "smoothing_width_s")),
            baseline_percentile=float(_require("metrics", data, "baseline_percentile")),
            sustained_threshold_sd=float(_require("metrics", data, "sustained_threshold_sd")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tr_seconds": self.tr_seconds,
            "instructional_window_s": (
                None if self.instructional_window_s is None else list(self.instructional_window_s)
            ),
            "smoothing_kernel": self.smoothing_kernel,
            "smoothing_width_s": self.smoothing_width_s,
            "baseline_percentile": self.baseline_percentile,
            "sustained_threshold_sd": self.sustained_threshold_sd,
        }


@dataclass(frozen=True)
class AnalysisConfig:
    """Parameters for §6.6 contrasts, §6.7 RSA and §8.2 standardization."""

    n_bootstrap: int
    n_permutations: int
    winsor_lower_pct: float
    winsor_upper_pct: float
    corpus_n_units: int

    def __post_init__(self) -> None:
        if self.n_bootstrap < 1:
            raise ConfigError(f"analysis.n_bootstrap must be >= 1; got {self.n_bootstrap}")
        if self.n_permutations < 1:
            raise ConfigError(f"analysis.n_permutations must be >= 1; got {self.n_permutations}")
        if not 0.0 <= self.winsor_lower_pct < self.winsor_upper_pct <= 100.0:
            raise ConfigError(
                "analysis winsor percentiles must satisfy 0 <= lower < upper <= 100; got "
                f"[{self.winsor_lower_pct}, {self.winsor_upper_pct}]"
            )
        if self.corpus_n_units < 1:
            raise ConfigError(f"analysis.corpus_n_units must be >= 1; got {self.corpus_n_units}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AnalysisConfig":
        allowed = [f.name for f in fields(cls)]
        _reject_unknown("analysis", data, allowed)
        return cls(
            n_bootstrap=int(_require("analysis", data, "n_bootstrap")),
            n_permutations=int(_require("analysis", data, "n_permutations")),
            winsor_lower_pct=float(_require("analysis", data, "winsor_lower_pct")),
            winsor_upper_pct=float(_require("analysis", data, "winsor_upper_pct")),
            corpus_n_units=int(_require("analysis", data, "corpus_n_units")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_bootstrap": self.n_bootstrap,
            "n_permutations": self.n_permutations,
            "winsor_lower_pct": self.winsor_lower_pct,
            "winsor_upper_pct": self.winsor_upper_pct,
            "corpus_n_units": self.corpus_n_units,
        }


@dataclass(frozen=True)
class TribeConfig:
    """Everything Phase II needs, loaded once from `config/tribe.yaml`.

    `project_root` is derived from the config file location (the parent of the
    directory holding the YAML) so that relative paths in the file resolve the
    same way on a laptop and on Colab.
    """

    checkpoint: str
    checkpoint_revision: str
    precision: str
    reading_rate_wpm: float
    atlas: str
    parcel_weighting: str
    batch_size: int
    vertex_retention_set: Path
    cache_root: Path
    atlas_file: Path
    checkpoints_lock: Path
    master_seed: int
    metrics: MetricsConfig
    analysis: AnalysisConfig
    project_root: Path = field(default_factory=Path)
    source_path: Optional[Path] = None

    _TOP_LEVEL_KEYS = (
        "checkpoint",
        "checkpoint_revision",
        "precision",
        "reading_rate_wpm",
        "atlas",
        "parcel_weighting",
        "batch_size",
        "vertex_retention_set",
        "cache_root",
        "atlas_file",
        "checkpoints_lock",
        "master_seed",
        "metrics",
        "analysis",
    )

    def __post_init__(self) -> None:
        if not self.checkpoint or "/" not in self.checkpoint:
            raise ConfigError(
                f"checkpoint must be a HuggingFace repo id like 'facebook/tribev2'; "
                f"got {self.checkpoint!r}"
            )
        if not _SHA_RE.match(self.checkpoint_revision):
            raise ConfigError(
                f"checkpoint_revision must be a 40-character lowercase hex commit SHA "
                f"(no default, no branch name, no tag); got {self.checkpoint_revision!r}"
            )
        if self.precision not in ALLOWED_PRECISION:
            raise ConfigError(
                f"precision must be one of {list(ALLOWED_PRECISION)}; got {self.precision!r}"
            )
        if float(self.reading_rate_wpm) not in ALLOWED_READING_RATES:
            raise ConfigError(
                f"reading_rate_wpm must be one of {list(ALLOWED_READING_RATES)} "
                f"(220 is the main specification, 180/260 are the §6.2 robustness values); "
                f"got {self.reading_rate_wpm!r}"
            )
        if self.atlas not in ALLOWED_ATLAS:
            raise ConfigError(f"atlas must be one of {list(ALLOWED_ATLAS)}; got {self.atlas!r}")
        if self.parcel_weighting not in ALLOWED_WEIGHTING:
            raise ConfigError(
                f"parcel_weighting must be one of {list(ALLOWED_WEIGHTING)}; "
                f"got {self.parcel_weighting!r}"
            )
        if self.batch_size < 1:
            raise ConfigError(f"batch_size must be >= 1; got {self.batch_size}")
        if self.master_seed < 0:
            raise ConfigError(f"master_seed must be >= 0; got {self.master_seed}")

    # -- derived ---------------------------------------------------------

    @property
    def revision_is_resolved(self) -> bool:
        return self.checkpoint_revision != UNRESOLVED_REVISION

    def require_resolved_revision(self) -> str:
        """Return the pinned revision, or raise if it is still the placeholder.

        Track A calls this before touching the network. Track B never needs it.
        """
        if not self.revision_is_resolved:
            raise ConfigError(
                f"checkpoint_revision in {self.source_path} is still the unresolved "
                f"placeholder ({UNRESOLVED_REVISION}). Resolve the real commit SHA of "
                f"'{self.checkpoint}' with "
                f"`python scripts/run_tribe_verification.py --config <cfg> --resolve-revision`, "
                f"write it into the config and into checkpoints.lock, and commit both. "
                f"Phase II refuses to run inference against an unpinned checkpoint (§6.1)."
            )
        return self.checkpoint_revision

    def identity(self) -> Dict[str, Any]:
        """The subset of config that participates in cache keys (§4.5)."""
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "precision": self.precision,
            "reading_rate_wpm": float(self.reading_rate_wpm),
            "atlas": self.atlas,
            "parcel_weighting": self.parcel_weighting,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "checkpoint_revision": self.checkpoint_revision,
            "precision": self.precision,
            "reading_rate_wpm": self.reading_rate_wpm,
            "atlas": self.atlas,
            "parcel_weighting": self.parcel_weighting,
            "batch_size": self.batch_size,
            "vertex_retention_set": str(self.vertex_retention_set),
            "cache_root": str(self.cache_root),
            "atlas_file": str(self.atlas_file),
            "checkpoints_lock": str(self.checkpoints_lock),
            "master_seed": self.master_seed,
            "metrics": self.metrics.as_dict(),
            "analysis": self.analysis.as_dict(),
        }

    def read_vertex_retention_set(self) -> Tuple[str, ...]:
        """Stimulus ids whose full `(T, V)` vertex array is kept permanently.

        One id per line; `#` comments and blank lines ignored. Missing file is
        an error, not an empty set -- an empty retention set silently discards
        every vertex array, which is unrecoverable (§4.5).
        """
        path = self.vertex_retention_set
        if not path.exists():
            raise ConfigError(
                f"vertex_retention_set file not found: {path}. This file decides which "
                f"vertex arrays survive; vertex data cannot be reconstructed from parcel "
                f"means, so a missing file is refused rather than treated as 'retain none'."
            )
        ids = []
        for line in path.read_text(encoding="utf-8").splitlines():
            entry = line.split("#", 1)[0].strip()
            if entry:
                ids.append(entry)
        return tuple(ids)


def load_config(path: str | Path, *, cache_root: str | Path | None = None) -> TribeConfig:
    """Load and validate `config/tribe.yaml`.

    `cache_root` overrides the value in the file. Colab passes the mounted
    Drive path there rather than editing a tracked config file per session.
    """
    import yaml

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level of the config must be a mapping")

    _reject_unknown("<top level>", raw, TribeConfig._TOP_LEVEL_KEYS)

    project_root = config_path.parent.parent

    def _resolve(value: Any) -> Path:
        candidate = Path(str(value)).expanduser()
        return candidate if candidate.is_absolute() else (project_root / candidate).resolve()

    metrics_raw = _require("<top level>", raw, "metrics")
    analysis_raw = _require("<top level>", raw, "analysis")
    if not isinstance(metrics_raw, dict) or not isinstance(analysis_raw, dict):
        raise ConfigError(f"{config_path}: 'metrics' and 'analysis' must be mappings")

    return TribeConfig(
        checkpoint=str(_require("<top level>", raw, "checkpoint")),
        checkpoint_revision=str(_require("<top level>", raw, "checkpoint_revision")),
        precision=str(_require("<top level>", raw, "precision")),
        reading_rate_wpm=float(_require("<top level>", raw, "reading_rate_wpm")),
        atlas=str(_require("<top level>", raw, "atlas")),
        parcel_weighting=str(_require("<top level>", raw, "parcel_weighting")),
        batch_size=int(_require("<top level>", raw, "batch_size")),
        vertex_retention_set=_resolve(_require("<top level>", raw, "vertex_retention_set")),
        cache_root=_resolve(
            cache_root if cache_root is not None else _require("<top level>", raw, "cache_root")
        ),
        atlas_file=_resolve(_require("<top level>", raw, "atlas_file")),
        checkpoints_lock=_resolve(_require("<top level>", raw, "checkpoints_lock")),
        master_seed=int(_require("<top level>", raw, "master_seed")),
        metrics=MetricsConfig.from_dict(metrics_raw),
        analysis=AnalysisConfig.from_dict(analysis_raw),
        project_root=project_root,
        source_path=config_path,
    )
