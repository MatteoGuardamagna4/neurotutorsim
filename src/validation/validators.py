"""
Deterministic answer validators for the NeuroTutorSim corpus.

Brief references:
  - §5.1.4  Validate all numeric answers with deterministic Python/R functions;
            store the validator name and expected output.
  - §5.4    Correctness via numeric execution or SymPy symbolic comparison.
  - §5.5    Reference correctness must be 100%; condition correctness >= 99%.

Design
------
Every unit in units.csv names a validator (column `validator`). The validator
recomputes the canonical answer from the unit's parameters, so `reference_answer`
and `transfer_answer` are never hand-trusted -- they are machine-recomputable and
reproducible. This is what decision gate 16 (§12.1) checks before the full corpus
is generated: build validators first, then 10 pilot units must pass.

Add a validator with the @register decorator; it becomes available by name via
get(name). Numeric validators compare with a tolerance; symbolic validators
compare with SymPy (imported lazily so the numeric path needs no SymPy install).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Callable, Dict

# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_REGISTRY: Dict[str, "Validator"] = {}


@dataclass(frozen=True)
class ValidationResult:
    validator: str
    expected: Any
    passed: bool
    detail: str = ""


class Validator:
    def __init__(self, name: str, fn: Callable[..., Any], kind: str) -> None:
        if kind not in ("numeric", "symbolic"):
            raise ValueError(f"kind must be 'numeric' or 'symbolic', got {kind!r}")
        self.name = name
        self.fn = fn
        self.kind = kind

    def expected(self, **params: Any) -> Any:
        """Recompute the canonical answer from the unit's parameters."""
        return self.fn(**params)

    def check(self, params: Dict[str, Any], answer: Any, tol: float = 1e-6) -> ValidationResult:
        expected = self.fn(**params)
        if self.kind == "numeric":
            passed = _numeric_close(answer, expected, tol)
            detail = f"answer={answer} expected={expected} tol={tol}"
        else:
            passed = _symbolic_equal(answer, expected)
            detail = f"answer={answer} expected={expected}"
        return ValidationResult(self.name, expected, passed, detail)


def register(name: str, kind: str = "numeric") -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _REGISTRY:
            raise ValueError(f"Validator {name!r} is already registered")
        _REGISTRY[name] = Validator(name, fn, kind)
        return fn
    return decorator


def get(name: str) -> Validator:
    if name not in _REGISTRY:
        raise KeyError(f"No validator named {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered() -> list[str]:
    return sorted(_REGISTRY)


# --------------------------------------------------------------------------- #
# Comparison helpers
# --------------------------------------------------------------------------- #

def _numeric_close(a: Any, b: Any, tol: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def _symbolic_equal(a: Any, b: Any) -> bool:
    import sympy as sp  # lazy: numeric path needs no SymPy install
    return bool(sp.simplify(sp.sympify(a) - sp.sympify(b)) == 0)


# --------------------------------------------------------------------------- #
# Domain 1 -- Bayesian Decision Analysis
# --------------------------------------------------------------------------- #

@register("bayes_posterior")
def bayes_posterior(*, prior: float, sensitivity: float, false_positive_rate: float) -> float:
    """P(H | positive evidence) for a binary hypothesis and binary test."""
    p_evidence = sensitivity * prior + false_positive_rate * (1 - prior)
    if p_evidence == 0:
        raise ValueError("P(evidence) is zero; check inputs")
    return sensitivity * prior / p_evidence


@register("expected_value")
def expected_value(*, payoffs: list[float], probs: list[float]) -> float:
    """Expected monetary value of a decision with discrete outcomes."""
    if len(payoffs) != len(probs):
        raise ValueError("payoffs and probs must be the same length")
    if not math.isclose(sum(probs), 1.0, abs_tol=1e-9):
        raise ValueError(f"probs must sum to 1, got {sum(probs)}")
    return sum(p * v for p, v in zip(probs, payoffs))


@register("evpi")
def evpi(*, payoffs_by_state: list[list[float]], state_probs: list[float]) -> float:
    """Expected Value of Perfect Information.

    payoffs_by_state[s][a] = payoff of action a if state s occurs.
    """
    n_actions = len(payoffs_by_state[0])
    ev_action = [
        sum(state_probs[s] * payoffs_by_state[s][a] for s in range(len(state_probs)))
        for a in range(n_actions)
    ]
    ev_best_action = max(ev_action)
    ev_with_info = sum(
        state_probs[s] * max(payoffs_by_state[s]) for s in range(len(state_probs))
    )
    return ev_with_info - ev_best_action


# --------------------------------------------------------------------------- #
# Domain 2 -- Causal Inference / A-B testing
# --------------------------------------------------------------------------- #

@register("two_proportion_z")
def two_proportion_z(*, x_a: int, n_a: int, x_b: int, n_b: int) -> float:
    """Pooled two-proportion z statistic for a conversion-rate A/B test."""
    p_a, p_b = x_a / n_a, x_b / n_b
    p_pool = (x_a + x_b) / (n_a + n_b)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se == 0:
        raise ValueError("standard error is zero; check inputs")
    return (p_b - p_a) / se


@register("ab_sample_size")
def ab_sample_size(*, baseline: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-group sample size for a two-proportion test (rounded up)."""
    p1 = baseline
    p2 = baseline + mde
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_beta = NormalDist().inv_cdf(power)
    numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(numerator / (p2 - p1) ** 2)


# --------------------------------------------------------------------------- #
# Symbolic example (demonstrates the SymPy path required by §5.4)
# --------------------------------------------------------------------------- #

@register("bernoulli_variance_expr", kind="symbolic")
def bernoulli_variance_expr(*, p_symbol: str = "p") -> str:
    """Canonical variance of a Bernoulli(p): p*(1-p)."""
    return f"{p_symbol}*(1-{p_symbol})"
