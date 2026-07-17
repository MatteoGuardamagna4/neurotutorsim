"""
Known-answer tests for the corpus validators.

These are the sanity checks §4.3 requires for scoring functions, and they double
as worked reference problems for the corpus. Run with:  python -m pytest tests/
"""

import math

from src.validation import validators as V


def test_registry_roundtrip():
    assert "bayes_posterior" in V.registered()
    with_error = False
    try:
        V.get("does_not_exist")
    except KeyError:
        with_error = True
    assert with_error


def test_bayes_posterior_disease_classic():
    # prior 1%, sensitivity 90%, false-positive 9%  ->  posterior ~ 0.0917
    result = V.get("bayes_posterior").check(
        params=dict(prior=0.01, sensitivity=0.90, false_positive_rate=0.09),
        answer=0.09174,
        tol=1e-4,
    )
    assert result.passed, result.detail


def test_expected_value():
    result = V.get("expected_value").check(
        params=dict(payoffs=[100, -50], probs=[0.6, 0.4]),
        answer=40.0,
    )
    assert result.passed, result.detail


def test_two_proportion_z():
    # 200/1000 vs 240/1000  ->  z ~ 2.159
    result = V.get("two_proportion_z").check(
        params=dict(x_a=200, n_a=1000, x_b=240, n_b=1000),
        answer=2.159,
        tol=1e-3,
    )
    assert result.passed, result.detail


def test_ab_sample_size_monotonic():
    # Smaller detectable effect must require a larger sample.
    n_big_effect = V.ab_sample_size(baseline=0.20, mde=0.05)
    n_small_effect = V.ab_sample_size(baseline=0.20, mde=0.02)
    assert n_small_effect > n_big_effect


def test_symbolic_path():
    result = V.get("bernoulli_variance_expr").check(
        params=dict(p_symbol="p"),
        answer="p - p**2",  # algebraically equal to p*(1-p)
    )
    assert result.passed, result.detail
