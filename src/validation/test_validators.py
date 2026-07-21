"""
Known-answer tests for the corpus validators.

These are the sanity checks §4.3 requires for scoring functions, and they double
as worked reference problems for the corpus. Run with:  python -m pytest
"""

import pytest

from src.validation import validators as V


def test_registry_roundtrip():
    assert "break_even_quantity" in V.registered()
    with_error = False
    try:
        V.get("does_not_exist")
    except KeyError:
        with_error = True
    assert with_error


def test_break_even_quantity():
    # FC 120,000 / (80 - 50) = 4,000 units
    result = V.get("break_even_quantity").check(
        params=dict(fixed_costs=120000, price=80, variable_cost=50),
        answer=4000,
    )
    assert result.passed, result.detail


def test_break_even_revenue():
    # 4,000 units * 80 = 320,000
    result = V.get("break_even_revenue").check(
        params=dict(fixed_costs=120000, price=80, variable_cost=50),
        answer=320000,
    )
    assert result.passed, result.detail


def test_target_profit_quantity():
    # (120,000 + 60,000) / (80 - 50) = 6,000 units
    result = V.get("target_profit_quantity").check(
        params=dict(fixed_costs=120000, target_profit=60000, price=80, variable_cost=50),
        answer=6000,
    )
    assert result.passed, result.detail


def test_price_for_break_even_quantity():
    # 90,000 / 3,000 + 12 = 42
    result = V.get("price_for_break_even_quantity").check(
        params=dict(fixed_costs=90000, quantity=3000, variable_cost=12),
        answer=42,
    )
    assert result.passed, result.detail


def test_contribution_margin_ratio():
    # (80 - 50) / 80 = 0.375
    result = V.get("contribution_margin_ratio").check(
        params=dict(price=80, variable_cost=50),
        answer=0.375,
    )
    assert result.passed, result.detail


@pytest.mark.parametrize(
    "validator_name,params",
    [
        ("break_even_quantity", dict(fixed_costs=120000, price=50, variable_cost=50)),
        ("break_even_revenue", dict(fixed_costs=120000, price=50, variable_cost=50)),
        ("target_profit_quantity", dict(fixed_costs=120000, target_profit=1000, price=50, variable_cost=50)),
        ("contribution_margin_ratio", dict(price=50, variable_cost=50)),
    ],
)
def test_raises_when_price_equals_variable_cost(validator_name, params):
    with pytest.raises(ValueError):
        V.get(validator_name).fn(**params)


def test_symbolic_path():
    result = V.get("contribution_margin_expr").check(
        params=dict(price_symbol="P", vc_symbol="V"),
        answer="P - V",
    )
    assert result.passed, result.detail
