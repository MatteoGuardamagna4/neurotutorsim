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
# Domain -- Managerial Accounting / Break-even analysis
# --------------------------------------------------------------------------- #

def _check_contribution_margin(price: float, variable_cost: float) -> None:
    if price <= variable_cost:
        raise ValueError(
            f"price ({price}) must exceed variable_cost ({variable_cost}); "
            "break-even is undefined when the contribution margin is non-positive"
        )


@register("break_even_quantity")
def break_even_quantity(*, fixed_costs: float, price: float, variable_cost: float) -> float:
    """Units required to break even: FC / (P - VC)."""
    if fixed_costs < 0:
        raise ValueError(f"fixed_costs must be >= 0, got {fixed_costs}")
    _check_contribution_margin(price, variable_cost)
    return fixed_costs / (price - variable_cost)


@register("break_even_revenue")
def break_even_revenue(*, fixed_costs: float, price: float, variable_cost: float) -> float:
    """Revenue at break-even: break_even_quantity * price."""
    return break_even_quantity(fixed_costs=fixed_costs, price=price, variable_cost=variable_cost) * price


@register("target_profit_quantity")
def target_profit_quantity(
    *, fixed_costs: float, target_profit: float, price: float, variable_cost: float
) -> float:
    """Units required to earn target_profit: (FC + target) / (P - VC)."""
    if fixed_costs < 0:
        raise ValueError(f"fixed_costs must be >= 0, got {fixed_costs}")
    _check_contribution_margin(price, variable_cost)
    return (fixed_costs + target_profit) / (price - variable_cost)


@register("price_for_break_even_quantity")
def price_for_break_even_quantity(*, fixed_costs: float, quantity: float, variable_cost: float) -> float:
    """Price that breaks even at a given quantity: FC / Q + VC."""
    if fixed_costs < 0:
        raise ValueError(f"fixed_costs must be >= 0, got {fixed_costs}")
    if quantity <= 0:
        raise ValueError(f"quantity must be > 0, got {quantity}")
    return fixed_costs / quantity + variable_cost


@register("contribution_margin_ratio")
def contribution_margin_ratio(*, price: float, variable_cost: float) -> float:
    """Contribution margin ratio: (P - VC) / P."""
    _check_contribution_margin(price, variable_cost)
    return (price - variable_cost) / price


# --------------------------------------------------------------------------- #
# Symbolic example (demonstrates the SymPy path required by §5.4)
# --------------------------------------------------------------------------- #

@register("contribution_margin_expr", kind="symbolic")
def contribution_margin_expr(*, price_symbol: str = "P", vc_symbol: str = "V") -> str:
    """Symbolic contribution margin: P - V."""
    return f"{price_symbol} - {vc_symbol}"
