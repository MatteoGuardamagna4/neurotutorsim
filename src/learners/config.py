"""Frozen configuration for Phase III (brief §7).

One YAML file (`config/learners.yaml`) drives the whole learner engine. The
dataclasses here are frozen and validate on construction: an unknown key, a
non-PSD correlation matrix, stratum weights that do not sum to one, or a
support level outside [0, 1] raises rather than being coerced to something
plausible. That strictness is the same precedent as `src/tribe/config.py`, and
for the same reason -- a run log records a config hash (§4.2), so a silently
accepted value would produce a silently wrong provenance record.

The low / medium / high convention
----------------------------------
§7.2 requires the baseline parameter values to be treated as assumptions, so
every free numeric parameter is declared in the YAML as a three-arm mapping::

    omega: {low: 1.5, medium: 2.5, high: 3.5}

`resolve_settings` walks the raw tree and collapses each such mapping to the
arm named by `parameter_setting`. A leaf that is *not* a `{low, medium, high}`
mapping is fixed at every setting -- which is how the values the brief fixes
(the 0.30/0.50/0.20 stratum weights, the eq. 15 state bounds) are protected
from a sensitivity sweep.

Track B. Imports numpy and pyyaml only; never torch.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

#: Order of the eq. (15) state vector. Every array of per-component values in
#: this package is in this order, and nothing may reorder it: the correlation
#: matrix rows in the YAML are read positionally against it.
STATE_COMPONENTS: Tuple[str, ...] = ("K", "M", "R", "C", "D")

#: Bounds fixed by eq. (15): every state component lives in [0, 1]. Not a
#: parameter, not swept, and never widened to hide a clipping problem.
STATE_LOWER = 0.0
STATE_UPPER = 1.0

ALLOWED_SETTINGS = ("low", "medium", "high")
ALLOWED_ENGINES = ("logistic", "minitaur")
ALLOWED_PERSISTENCE = ("immediate_withdrawal", "gradual_fading", "persistent")

#: Instructional conditions (§3.2). `static_ai` is the optional fourth arm.
TRADITIONAL = "traditional"
AI_SCAFFOLDING = "ai_scaffolding"
AI_SUBSTITUTION = "ai_substitution"
STATIC_AI = "static_ai"
ALLOWED_CONDITIONS = (TRADITIONAL, AI_SCAFFOLDING, AI_SUBSTITUTION, STATIC_AI)

#: Named per-stream RNG substreams (§10.1). Fixed set: a stream that appears in
#: the config but is never consumed, or vice versa, is a reproducibility bug.
REQUIRED_STREAMS = ("population", "responses", "tutor", "checkpoints")

_SWEEP_KEYS = frozenset(ALLOWED_SETTINGS)


class LearnerConfigError(ValueError):
    """Raised for any malformed or unknown Phase III configuration entry."""


# ---------------------------------------------------------------------------
# Setting resolution
# ---------------------------------------------------------------------------


def is_sweep(node: Any) -> bool:
    """Is `node` a `{low, medium, high}` sweep mapping?

    The test is exact-key equality, not membership: a mapping with only
    `{low, high}` is a typo (a missing baseline), and a mapping that merely
    happens to contain a `low` key -- like a per-stratum table -- is not a
    sweep.
    """
    return isinstance(node, Mapping) and set(node) == _SWEEP_KEYS


def resolve_settings(node: Any, setting: str, *, path: str = "") -> Any:
    """Collapse every sweep mapping under `node` to its `setting` arm.

    Parameters
    ----------
    node
        Raw YAML subtree.
    setting
        One of `low`, `medium`, `high`.
    path
        Dotted key path, used only to make an error message locatable.

    Returns
    -------
    Any
        The subtree with every sweep mapping replaced by one arm.
    """
    if setting not in ALLOWED_SETTINGS:
        raise LearnerConfigError(
            f"parameter_setting must be one of {list(ALLOWED_SETTINGS)}; got {setting!r}"
        )
    if is_sweep(node):
        return resolve_settings(node[setting], setting, path=f"{path}[{setting}]")
    if isinstance(node, Mapping):
        return {
            key: resolve_settings(value, setting, path=f"{path}.{key}" if path else str(key))
            for key, value in node.items()
        }
    if isinstance(node, (list, tuple)):
        return [resolve_settings(item, setting, path=f"{path}[{i}]") for i, item in enumerate(node)]
    return node


# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _reject_unknown(section: str, data: Mapping[str, Any], allowed: Sequence[str]) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise LearnerConfigError(
            f"unknown key(s) in config section {section!r}: {unknown}. "
            f"Allowed keys: {sorted(allowed)}. Configuration is not coerced -- "
            f"add the field to the dataclass or fix the spelling."
        )


def _require(section: str, data: Mapping[str, Any], key: str) -> Any:
    if key not in data:
        raise LearnerConfigError(f"config section {section!r} is missing required key {key!r}")
    return data[key]


def _unit_interval(section: str, name: str, value: float) -> float:
    number = float(value)
    if not STATE_LOWER <= number <= STATE_UPPER:
        raise LearnerConfigError(
            f"{section}.{name} must lie in [{STATE_LOWER}, {STATE_UPPER}]; got {number}"
        )
    return number


def _positive(section: str, name: str, value: float) -> float:
    number = float(value)
    if not number > STATE_LOWER:
        raise LearnerConfigError(f"{section}.{name} must be > 0; got {number}")
    return number


def _ladder(section: str, name: str, value: Any) -> Tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise LearnerConfigError(
            f"{section}.{name} must be a non-empty list of support levels; got {value!r}"
        )
    levels = tuple(_unit_interval(section, name, item) for item in value)
    if list(levels) != sorted(levels):
        raise LearnerConfigError(
            f"{section}.{name} must be non-decreasing -- a hint ladder that gets *less* "
            f"supportive with depth is a specification error, not a policy; got {list(levels)}"
        )
    return levels


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PopulationConfig:
    """§7.1-7.2 learner initialisation, eq. (15) state and eq. (16) parameters."""

    n_learners: int
    stratum_names: Tuple[str, ...]
    stratum_weights: Tuple[float, ...]
    state_means: Tuple[float, ...]
    state_sds: Tuple[float, ...]
    stratum_k_shift: Tuple[float, ...]
    stratum_m_shift: Tuple[float, ...]
    correlation: Tuple[Tuple[float, ...], ...]
    mu_alpha: float
    sigma_alpha: float
    a_delta: float
    b_delta: float
    theta_intercept: float
    theta_slope: float
    theta_center: float
    confidence_bias_mu: float
    confidence_bias_sd: float
    speed_sigma: float

    def __post_init__(self) -> None:
        section = "population"
        if self.n_learners < 1:
            raise LearnerConfigError(f"{section}.n_learners must be >= 1; got {self.n_learners}")
        n_strata = len(self.stratum_names)
        if n_strata < 1:
            raise LearnerConfigError(f"{section}.stratum_names must not be empty")
        for name, values in (
            ("stratum_weights", self.stratum_weights),
            ("stratum_k_shift", self.stratum_k_shift),
            ("stratum_m_shift", self.stratum_m_shift),
        ):
            if len(values) != n_strata:
                raise LearnerConfigError(
                    f"{section}.{name} has {len(values)} entries but there are {n_strata} "
                    f"strata ({list(self.stratum_names)}); the lists are read positionally"
                )
        total = float(sum(self.stratum_weights))
        if not np.isclose(total, STATE_UPPER):
            raise LearnerConfigError(
                f"{section}.stratum_weights must sum to 1; got {total} for "
                f"{list(self.stratum_weights)}"
            )
        if any(weight < STATE_LOWER for weight in self.stratum_weights):
            raise LearnerConfigError(
                f"{section}.stratum_weights must be non-negative; got {list(self.stratum_weights)}"
            )
        n_state = len(STATE_COMPONENTS)
        if len(self.state_means) != n_state or len(self.state_sds) != n_state:
            raise LearnerConfigError(
                f"{section}.state_means and state_sds must each have one entry per state "
                f"component {list(STATE_COMPONENTS)}"
            )
        for value in self.state_sds:
            _positive(section, "state_sds", value)
        matrix = np.asarray(self.correlation, dtype=np.float64)
        if matrix.shape != (n_state, n_state):
            raise LearnerConfigError(
                f"{section}.correlation must be {n_state}x{n_state} for state components "
                f"{list(STATE_COMPONENTS)}; got shape {matrix.shape}"
            )
        if not np.allclose(matrix, matrix.T):
            raise LearnerConfigError(f"{section}.correlation must be symmetric")
        if not np.allclose(np.diag(matrix), STATE_UPPER):
            raise LearnerConfigError(
                f"{section}.correlation must have a unit diagonal; got {np.diag(matrix).tolist()}"
            )
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues.min() < -np.sqrt(np.finfo(np.float64).eps):
            raise LearnerConfigError(
                f"{section}.correlation is not positive semi-definite (smallest eigenvalue "
                f"{eigenvalues.min():.6g}). A Gaussian copula needs a valid correlation "
                f"matrix; it is not repaired or nudged here."
            )
        _positive(section, "sigma_alpha", self.sigma_alpha)
        _positive(section, "a_delta", self.a_delta)
        _positive(section, "b_delta", self.b_delta)
        _positive(section, "theta_slope", self.theta_slope)
        _positive(section, "speed_sigma", self.speed_sigma)
        _positive(section, "confidence_bias_sd", self.confidence_bias_sd)
        _unit_interval(section, "theta_center", self.theta_center)

    @property
    def n_strata(self) -> int:
        return len(self.stratum_names)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PopulationConfig":
        section = "population"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        means = _require(section, data, "state_means")
        sds = _require(section, data, "state_sds")
        for name, table in (("state_means", means), ("state_sds", sds)):
            if not isinstance(table, Mapping) or set(table) != set(STATE_COMPONENTS):
                raise LearnerConfigError(
                    f"{section}.{name} must be a mapping with exactly the keys "
                    f"{list(STATE_COMPONENTS)}; got {sorted(table) if isinstance(table, Mapping) else table!r}"
                )
        return cls(
            n_learners=int(_require(section, data, "n_learners")),
            stratum_names=tuple(str(x) for x in _require(section, data, "stratum_names")),
            stratum_weights=tuple(float(x) for x in _require(section, data, "stratum_weights")),
            state_means=tuple(
                _unit_interval(section, f"state_means.{key}", means[key])
                for key in STATE_COMPONENTS
            ),
            state_sds=tuple(float(sds[key]) for key in STATE_COMPONENTS),
            stratum_k_shift=tuple(float(x) for x in _require(section, data, "stratum_k_shift")),
            stratum_m_shift=tuple(float(x) for x in _require(section, data, "stratum_m_shift")),
            correlation=tuple(
                tuple(float(x) for x in row) for row in _require(section, data, "correlation")
            ),
            mu_alpha=float(_require(section, data, "mu_alpha")),
            sigma_alpha=float(_require(section, data, "sigma_alpha")),
            a_delta=float(_require(section, data, "a_delta")),
            b_delta=float(_require(section, data, "b_delta")),
            theta_intercept=float(_require(section, data, "theta_intercept")),
            theta_slope=float(_require(section, data, "theta_slope")),
            theta_center=float(_require(section, data, "theta_center")),
            confidence_bias_mu=float(_require(section, data, "confidence_bias_mu")),
            confidence_bias_sd=float(_require(section, data, "confidence_bias_sd")),
            speed_sigma=float(_require(section, data, "speed_sigma")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_learners": self.n_learners,
            "stratum_names": list(self.stratum_names),
            "stratum_weights": list(self.stratum_weights),
            "state_means": dict(zip(STATE_COMPONENTS, self.state_means)),
            "state_sds": dict(zip(STATE_COMPONENTS, self.state_sds)),
            "stratum_k_shift": list(self.stratum_k_shift),
            "stratum_m_shift": list(self.stratum_m_shift),
            "correlation": [list(row) for row in self.correlation],
            "mu_alpha": self.mu_alpha,
            "sigma_alpha": self.sigma_alpha,
            "a_delta": self.a_delta,
            "b_delta": self.b_delta,
            "theta_intercept": self.theta_intercept,
            "theta_slope": self.theta_slope,
            "theta_center": self.theta_center,
            "confidence_bias_mu": self.confidence_bias_mu,
            "confidence_bias_sd": self.confidence_bias_sd,
            "speed_sigma": self.speed_sigma,
        }


@dataclass(frozen=True)
class CurriculumConfig:
    """Difficulty -> logit-scale `b_u` map, and the decision-gate-16 threshold."""

    b_intercept: float
    b_slope: float
    difficulty_center: float
    difficulty_min: int
    difficulty_max: int
    min_units: int

    def __post_init__(self) -> None:
        section = "curriculum"
        if self.b_slope <= STATE_LOWER:
            raise LearnerConfigError(
                f"{section}.b_slope must be > 0: the difficulty -> b_u map is required to be "
                f"monotone increasing, so that a harder unit is a harder unit; got {self.b_slope}"
            )
        if self.difficulty_min >= self.difficulty_max:
            raise LearnerConfigError(
                f"{section}.difficulty_min must be < difficulty_max; got "
                f"[{self.difficulty_min}, {self.difficulty_max}]"
            )
        if not self.difficulty_min <= self.difficulty_center <= self.difficulty_max:
            raise LearnerConfigError(
                f"{section}.difficulty_center must lie inside "
                f"[{self.difficulty_min}, {self.difficulty_max}]; got {self.difficulty_center}"
            )
        if self.min_units < 1:
            raise LearnerConfigError(f"{section}.min_units must be >= 1; got {self.min_units}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CurriculumConfig":
        section = "curriculum"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(
            b_intercept=float(_require(section, data, "b_intercept")),
            b_slope=float(_require(section, data, "b_slope")),
            difficulty_center=float(_require(section, data, "difficulty_center")),
            difficulty_min=int(_require(section, data, "difficulty_min")),
            difficulty_max=int(_require(section, data, "difficulty_max")),
            min_units=int(_require(section, data, "min_units")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class ResponseConfig:
    """§7.3 eq. (17)-(18) coefficients plus the confidence, request and latency models."""

    rho: float
    kappa: float
    omega: float
    near_b_delta: float
    far_b_delta: float
    transfer_b_delta: float
    confidence_noise_sd: float
    request_intercept: float
    request_dependence_slope: float
    request_ability_slope: float
    latency_base_s: float
    latency_per_hint_s: float
    latency_attempt_s: float

    def __post_init__(self) -> None:
        section = "response"
        if self.omega <= STATE_LOWER:
            raise LearnerConfigError(
                f"{section}.omega must be > 0: eq. (18) adds omega*h to the eq. (17) logit, so a "
                f"non-positive omega would make support harmful or inert by construction, which "
                f"is a modelling claim and not a parameter value; got {self.omega}"
            )
        _positive(section, "confidence_noise_sd", self.confidence_noise_sd)
        for name in ("near_b_delta", "far_b_delta", "transfer_b_delta"):
            if getattr(self, name) < STATE_LOWER:
                raise LearnerConfigError(
                    f"{section}.{name} must be >= 0 -- a transfer probe is not easier than the "
                    f"trained item; got {getattr(self, name)}"
                )
        if self.far_b_delta < self.near_b_delta:
            raise LearnerConfigError(
                f"{section}.far_b_delta must be >= near_b_delta; got {self.far_b_delta} < "
                f"{self.near_b_delta}"
            )
        for name in ("latency_base_s", "latency_per_hint_s", "latency_attempt_s"):
            if getattr(self, name) < STATE_LOWER:
                raise LearnerConfigError(f"{section}.{name} must be >= 0; got {getattr(self, name)}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResponseConfig":
        section = "response"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(**{f.name: float(_require(section, data, f.name)) for f in fields(cls)})

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class SupportConfig:
    """§3.4 deterministic tutor policies and the support-persistence modifier."""

    hint_ladder_h: Tuple[float, ...]
    scaffolding_h: Tuple[float, ...]
    substitution_h: float
    static_ai_h: float
    adaptation_traditional: float
    adaptation_static_ai: float
    adaptation_scaffolding_base: float
    adaptation_scaffolding_gain: float
    adaptation_substitution: float
    diagnosis_accuracy: float
    mismatch_scale: float
    persistence_policy: str
    fade_base: float
    withdrawal_success_threshold: int

    def __post_init__(self) -> None:
        section = "support"
        # Validated here rather than only in `from_dict`, so the guard holds
        # however the object is constructed -- `dataclasses.replace` in a test or
        # a sensitivity sweep goes through `__post_init__` but not `from_dict`.
        for name in ("hint_ladder_h", "scaffolding_h"):
            _ladder(section, name, list(getattr(self, name)))
        if len(self.hint_ladder_h) != len(self.scaffolding_h):
            raise LearnerConfigError(
                f"{section}.hint_ladder_h and scaffolding_h must have the same depth so that a "
                f"condition contrast is not also a ladder-length contrast; got "
                f"{len(self.hint_ladder_h)} and {len(self.scaffolding_h)}"
            )
        _unit_interval(section, "substitution_h", self.substitution_h)
        _unit_interval(section, "static_ai_h", self.static_ai_h)
        for name in (
            "adaptation_traditional",
            "adaptation_static_ai",
            "adaptation_scaffolding_base",
            "adaptation_substitution",
            "diagnosis_accuracy",
            "fade_base",
        ):
            _unit_interval(section, name, getattr(self, name))
        ceiling = self.adaptation_scaffolding_base + self.adaptation_scaffolding_gain
        if not STATE_LOWER <= ceiling <= STATE_UPPER:
            raise LearnerConfigError(
                f"{section}.adaptation_scaffolding_base + adaptation_scaffolding_gain must lie "
                f"in [{STATE_LOWER}, {STATE_UPPER}] -- adaptation is a proportion, and it is not "
                f"clipped silently; got {ceiling}"
            )
        _positive(section, "mismatch_scale", self.mismatch_scale)
        if self.persistence_policy not in ALLOWED_PERSISTENCE:
            raise LearnerConfigError(
                f"{section}.persistence_policy must be one of {list(ALLOWED_PERSISTENCE)}; "
                f"got {self.persistence_policy!r}"
            )
        if self.withdrawal_success_threshold < 1:
            raise LearnerConfigError(
                f"{section}.withdrawal_success_threshold must be >= 1; got "
                f"{self.withdrawal_success_threshold}"
            )

    @property
    def max_hint_depth(self) -> int:
        """Depth of the prewritten ladder -- the maximum number of hints given."""
        return len(self.hint_ladder_h)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SupportConfig":
        section = "support"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(
            hint_ladder_h=_ladder(section, "hint_ladder_h", _require(section, data, "hint_ladder_h")),
            scaffolding_h=_ladder(section, "scaffolding_h", _require(section, data, "scaffolding_h")),
            substitution_h=float(_require(section, data, "substitution_h")),
            static_ai_h=float(_require(section, data, "static_ai_h")),
            adaptation_traditional=float(_require(section, data, "adaptation_traditional")),
            adaptation_static_ai=float(_require(section, data, "adaptation_static_ai")),
            adaptation_scaffolding_base=float(
                _require(section, data, "adaptation_scaffolding_base")
            ),
            adaptation_scaffolding_gain=float(
                _require(section, data, "adaptation_scaffolding_gain")
            ),
            adaptation_substitution=float(_require(section, data, "adaptation_substitution")),
            diagnosis_accuracy=float(_require(section, data, "diagnosis_accuracy")),
            mismatch_scale=float(_require(section, data, "mismatch_scale")),
            persistence_policy=str(_require(section, data, "persistence_policy")),
            fade_base=float(_require(section, data, "fade_base")),
            withdrawal_success_threshold=int(
                _require(section, data, "withdrawal_success_threshold")
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["hint_ladder_h"] = list(self.hint_ladder_h)
        out["scaffolding_h"] = list(self.scaffolding_h)
        return out


@dataclass(frozen=True)
class EffortConfig:
    """§7.4 eq. (19) coefficients, plus the §10.3 zero-effort control override."""

    a0: float
    a1: float
    a2: float
    a3: float
    a4: float
    zero_effort_override: Tuple[Tuple[str, float], ...]

    _COEFFICIENTS = ("a0", "a1", "a2", "a3", "a4")

    def __post_init__(self) -> None:
        section = "effort"
        for name, _ in self.zero_effort_override:
            if name not in self._COEFFICIENTS:
                raise LearnerConfigError(
                    f"{section}.zero_effort_override may only name eq. (19) coefficients "
                    f"{list(self._COEFFICIENTS)}; got {name!r}"
                )

    def zeroed(self) -> "EffortConfig":
        """The §10.3 zero-effort control: this config with the override applied.

        Every coefficient the override names is replaced. Nothing else moves,
        so the control differs from the baseline in exactly the declared way.
        """
        return replace(self, **dict(self.zero_effort_override))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffortConfig":
        section = "effort"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        override = _require(section, data, "zero_effort_override")
        if not isinstance(override, Mapping):
            raise LearnerConfigError(
                f"{section}.zero_effort_override must be a mapping of coefficient name to value; "
                f"got {override!r}"
            )
        return cls(
            a0=float(_require(section, data, "a0")),
            a1=float(_require(section, data, "a1")),
            a2=float(_require(section, data, "a2")),
            a3=float(_require(section, data, "a3")),
            a4=float(_require(section, data, "a4")),
            zero_effort_override=tuple(
                (str(key), float(value)) for key, value in sorted(override.items())
            ),
        )

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {name: getattr(self, name) for name in self._COEFFICIENTS}
        out["zero_effort_override"] = dict(self.zero_effort_override)
        return out


@dataclass(frozen=True)
class EffectivenessConfig:
    """§7.4 eq. (20) coefficients and the corpus-column fallbacks."""

    f0: float
    f1: float
    f2: float
    f3: float
    f4: float
    coverage_default: float
    correctness_default: float

    def __post_init__(self) -> None:
        section = "effectiveness"
        _unit_interval(section, "coverage_default", self.coverage_default)
        _unit_interval(section, "correctness_default", self.correctness_default)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EffectivenessConfig":
        section = "effectiveness"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(**{f.name: float(_require(section, data, f.name)) for f in fields(cls)})

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class UpdateConfig:
    """§7.6 eq. (21)-(25) coefficients.

    eq. (21) carries no coefficient of its own -- it uses the per-learner
    `alpha_i` and `delta_i` of eq. (16) -- so none is exposed here.
    """

    m_decay_scale: float
    eta_M: float
    eta_C: float
    eta_R: float
    eta_O: float
    brier_max: float
    eta_D: float
    eta_F: float

    def __post_init__(self) -> None:
        section = "updates"
        if self.m_decay_scale < STATE_LOWER:
            raise LearnerConfigError(
                f"{section}.m_decay_scale must be >= 0; got {self.m_decay_scale}"
            )
        for name in ("eta_M", "eta_C", "eta_R", "eta_O", "eta_D", "eta_F"):
            if getattr(self, name) < STATE_LOWER:
                raise LearnerConfigError(
                    f"{section}.{name} must be >= 0. The sign of each term is fixed by the "
                    f"equation it appears in (eq. 22, 23, 25); flipping it here would silently "
                    f"reverse the direction of a state update. Got {getattr(self, name)}"
                )
        _positive(section, "brier_max", self.brier_max)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UpdateConfig":
        section = "updates"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(**{f.name: float(_require(section, data, f.name)) for f in fields(cls)})

    def as_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True)
class CheckpointConfig:
    """§7.7 assessment schedule and probe composition."""

    episodes: Tuple[int, ...]
    retention_interval_episodes: int
    n_trained_items: int
    n_near_items: int
    n_far_items: int
    probe_h: float
    ece_bins: int
    separation_threshold: float

    def __post_init__(self) -> None:
        section = "checkpoints"
        if not self.episodes:
            raise LearnerConfigError(f"{section}.episodes must list at least one episode index")
        if any(index < 0 for index in self.episodes):
            raise LearnerConfigError(
                f"{section}.episodes are 0-based indices and must be >= 0; got {list(self.episodes)}"
            )
        if list(self.episodes) != sorted(set(self.episodes)):
            raise LearnerConfigError(
                f"{section}.episodes must be strictly increasing and unique; got "
                f"{list(self.episodes)}"
            )
        if self.retention_interval_episodes < 1:
            raise LearnerConfigError(
                f"{section}.retention_interval_episodes must be >= 1 -- a retention probe with a "
                f"zero interval measures immediate accuracy, not retention; got "
                f"{self.retention_interval_episodes}"
            )
        for name in ("n_trained_items", "n_near_items", "n_far_items", "ece_bins"):
            if getattr(self, name) < 1:
                raise LearnerConfigError(f"{section}.{name} must be >= 1; got {getattr(self, name)}")
        _unit_interval(section, "probe_h", self.probe_h)
        if self.probe_h <= STATE_LOWER:
            raise LearnerConfigError(
                f"{section}.probe_h must be > 0: eq. (26) is a supported-minus-unaided "
                f"difference, which is identically zero at h = 0; got {self.probe_h}"
            )
        if self.separation_threshold < STATE_LOWER:
            raise LearnerConfigError(
                f"{section}.separation_threshold must be >= 0; got {self.separation_threshold}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointConfig":
        section = "checkpoints"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(
            episodes=tuple(int(x) for x in _require(section, data, "episodes")),
            retention_interval_episodes=int(
                _require(section, data, "retention_interval_episodes")
            ),
            n_trained_items=int(_require(section, data, "n_trained_items")),
            n_near_items=int(_require(section, data, "n_near_items")),
            n_far_items=int(_require(section, data, "n_far_items")),
            probe_h=float(_require(section, data, "probe_h")),
            ece_bins=int(_require(section, data, "ece_bins")),
            separation_threshold=float(_require(section, data, "separation_threshold")),
        )

    def as_dict(self) -> Dict[str, Any]:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out["episodes"] = list(self.episodes)
        return out


@dataclass(frozen=True)
class SeedConfig:
    """§10.1 seeding. One master seed, named substreams via `SeedSequence.spawn`."""

    master: int
    streams: Tuple[str, ...]

    def __post_init__(self) -> None:
        section = "seeds"
        if self.master < 0:
            raise LearnerConfigError(f"{section}.master must be >= 0; got {self.master}")
        if tuple(self.streams) != REQUIRED_STREAMS:
            raise LearnerConfigError(
                f"{section}.streams must be exactly {list(REQUIRED_STREAMS)} in that order. The "
                f"stream order fixes which spawned SeedSequence each consumer gets, so adding, "
                f"removing or reordering a name changes every draw in the simulation. "
                f"Got {list(self.streams)}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SeedConfig":
        section = "seeds"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(
            master=int(_require(section, data, "master")),
            streams=tuple(str(x) for x in _require(section, data, "streams")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {"master": self.master, "streams": list(self.streams)}


@dataclass(frozen=True)
class EngineConfig:
    """§10.2 response-engine selection. Swapping engines is this one field."""

    name: str

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_ENGINES:
            raise LearnerConfigError(
                f"engine.name must be one of {list(ALLOWED_ENGINES)}; got {self.name!r}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EngineConfig":
        _reject_unknown("engine", data, [f.name for f in fields(cls)])
        return cls(name=str(_require("engine", data, "name")))

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name}


@dataclass(frozen=True)
class RunConfig:
    """Run scope: episode count, conditions, and the output chunk size."""

    episodes: int
    conditions: Tuple[str, ...]
    chunk_episodes: int

    def __post_init__(self) -> None:
        section = "run"
        if self.episodes < 1:
            raise LearnerConfigError(f"{section}.episodes must be >= 1; got {self.episodes}")
        if not self.conditions:
            raise LearnerConfigError(f"{section}.conditions must name at least one condition")
        unknown = [c for c in self.conditions if c not in ALLOWED_CONDITIONS]
        if unknown:
            raise LearnerConfigError(
                f"{section}.conditions contains unknown condition(s) {unknown}; allowed: "
                f"{list(ALLOWED_CONDITIONS)}"
            )
        if len(set(self.conditions)) != len(self.conditions):
            raise LearnerConfigError(f"{section}.conditions must be unique; got {list(self.conditions)}")
        if self.chunk_episodes < 1:
            raise LearnerConfigError(
                f"{section}.chunk_episodes must be >= 1; got {self.chunk_episodes}"
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunConfig":
        section = "run"
        _reject_unknown(section, data, [f.name for f in fields(cls)])
        return cls(
            episodes=int(_require(section, data, "episodes")),
            conditions=tuple(str(x) for x in _require(section, data, "conditions")),
            chunk_episodes=int(_require(section, data, "chunk_episodes")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "conditions": list(self.conditions),
            "chunk_episodes": self.chunk_episodes,
        }


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnerConfig:
    """Everything Phase III needs, loaded once from `config/learners.yaml`."""

    parameter_setting: str
    population: PopulationConfig
    curriculum: CurriculumConfig
    response: ResponseConfig
    support: SupportConfig
    effort: EffortConfig
    effectiveness: EffectivenessConfig
    updates: UpdateConfig
    checkpoints: CheckpointConfig
    seeds: SeedConfig
    engine: EngineConfig
    run: RunConfig
    project_root: Path = Path()
    source_path: Optional[Path] = None

    _TOP_LEVEL_KEYS = (
        "parameter_setting",
        "population",
        "curriculum",
        "response",
        "support",
        "effort",
        "effectiveness",
        "updates",
        "checkpoints",
        "seeds",
        "engine",
        "run",
    )

    def __post_init__(self) -> None:
        if self.parameter_setting not in ALLOWED_SETTINGS:
            raise LearnerConfigError(
                f"parameter_setting must be one of {list(ALLOWED_SETTINGS)}; "
                f"got {self.parameter_setting!r}"
            )

    def as_dict(self) -> Dict[str, Any]:
        """Config as plain data. This is what the run log and the hash see."""
        return {
            "parameter_setting": self.parameter_setting,
            "population": self.population.as_dict(),
            "curriculum": self.curriculum.as_dict(),
            "response": self.response.as_dict(),
            "support": self.support.as_dict(),
            "effort": self.effort.as_dict(),
            "effectiveness": self.effectiveness.as_dict(),
            "updates": self.updates.as_dict(),
            "checkpoints": self.checkpoints.as_dict(),
            "seeds": self.seeds.as_dict(),
            "engine": self.engine.as_dict(),
            "run": self.run.as_dict(),
        }

    def config_hash(self) -> str:
        """sha256 over the canonical JSON of `as_dict()`.

        `hashlib`, not Python's `hash()`: the built-in hash of a string is salted
        per process (PYTHONHASHSEED) and would differ between the two runs a
        determinism check compares.
        """
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_learner_config(
    path: str | Path,
    *,
    setting: Optional[str] = None,
    n_learners: Optional[int] = None,
    episodes: Optional[int] = None,
    conditions: Optional[Sequence[str]] = None,
    engine: Optional[str] = None,
) -> LearnerConfig:
    """Load and validate `config/learners.yaml`.

    Parameters
    ----------
    path
        Path to the YAML file.
    setting
        Overrides the file's `parameter_setting` (§7.2 sensitivity arm).
    n_learners, episodes, conditions, engine
        CLI overrides applied after validation, so an override is validated by
        the same dataclass as a file value.
    """
    import yaml

    config_path = Path(path).resolve()
    if not config_path.exists():
        raise LearnerConfigError(f"config file not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LearnerConfigError(f"{config_path}: top level of the config must be a mapping")
    _reject_unknown("<top level>", raw, LearnerConfig._TOP_LEVEL_KEYS)

    active = str(setting or _require("<top level>", raw, "parameter_setting"))
    resolved = resolve_settings(raw, active)

    for group in LearnerConfig._TOP_LEVEL_KEYS:
        if group == "parameter_setting":
            continue
        if not isinstance(resolved.get(group), Mapping):
            raise LearnerConfigError(
                f"{config_path}: config group {group!r} must be a mapping; got "
                f"{resolved.get(group)!r}"
            )

    config = LearnerConfig(
        parameter_setting=active,
        population=PopulationConfig.from_dict(resolved["population"]),
        curriculum=CurriculumConfig.from_dict(resolved["curriculum"]),
        response=ResponseConfig.from_dict(resolved["response"]),
        support=SupportConfig.from_dict(resolved["support"]),
        effort=EffortConfig.from_dict(resolved["effort"]),
        effectiveness=EffectivenessConfig.from_dict(resolved["effectiveness"]),
        updates=UpdateConfig.from_dict(resolved["updates"]),
        checkpoints=CheckpointConfig.from_dict(resolved["checkpoints"]),
        seeds=SeedConfig.from_dict(resolved["seeds"]),
        engine=EngineConfig.from_dict(resolved["engine"]),
        run=RunConfig.from_dict(resolved["run"]),
        project_root=config_path.parent.parent,
        source_path=config_path,
    )

    if n_learners is not None:
        config = replace(config, population=replace(config.population, n_learners=int(n_learners)))
    if episodes is not None:
        config = replace(config, run=replace(config.run, episodes=int(episodes)))
    if conditions is not None:
        config = replace(
            config, run=replace(config.run, conditions=tuple(str(c) for c in conditions))
        )
    if engine is not None:
        config = replace(config, engine=EngineConfig(name=str(engine)))
    return config
