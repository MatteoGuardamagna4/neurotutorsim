"""Neural metrics against analytically known inputs (§6.5, decision gate 19)."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import neural_metrics as M
from tests.conftest import make_parcel_frame

NETWORKS = {1: "A", 2: "A", 3: "B", 4: "B"}


def test_mean_response_is_the_arithmetic_mean():
    assert M.mean_response([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_auc_of_a_triangle_wave_is_its_area():
    # 0 -> 1 -> 0 over 2 seconds at TR=1: two triangles, total area 1.0
    assert M.auc([0.0, 1.0, 0.0], tr_seconds=1.0) == pytest.approx(1.0)


def test_auc_scales_linearly_with_tr():
    series = [0.0, 1.0, 0.0]
    assert M.auc(series, tr_seconds=2.0) == pytest.approx(2.0)


def test_auc_of_a_constant_is_height_times_duration():
    # 5 samples at TR=1 spans 4 seconds between first and last sample
    assert M.auc([3.0] * 5, tr_seconds=1.0) == pytest.approx(12.0)


def test_auc_rejects_a_non_positive_tr():
    with pytest.raises(ValueError, match="tr_seconds"):
        M.auc([1.0, 2.0], tr_seconds=0.0)


def test_peak_response_is_the_maximum_when_smoothing_is_a_no_op():
    series = [0.0, 5.0, 1.0, 2.0]
    assert M.peak_response(series, "boxcar", width_s=1.0, tr_seconds=1.0) == pytest.approx(5.0)


def test_smoothing_flattens_a_single_spike():
    series = [0.0, 0.0, 9.0, 0.0, 0.0]
    smoothed = M.peak_response(series, "boxcar", width_s=3.0, tr_seconds=1.0)
    assert smoothed == pytest.approx(3.0)  # (0 + 9 + 0) / 3


def test_boxcar_smoothing_preserves_a_constant_at_the_edges():
    smoothed = M.smooth([2.0] * 5, "boxcar", width_s=3.0, tr_seconds=1.0)
    assert smoothed == pytest.approx([2.0] * 5)


def test_unknown_kernel_raises():
    with pytest.raises(ValueError, match="unknown smoothing kernel"):
        M.smooth([1.0, 2.0], "triangular", width_s=1.0, tr_seconds=1.0)


def test_time_to_peak_is_in_seconds():
    series = [0.0, 1.0, 7.0, 2.0]
    assert M.time_to_peak(series, "boxcar", 1.0, tr_seconds=1.0) == pytest.approx(2.0)
    assert M.time_to_peak(series, "boxcar", 1.0, tr_seconds=2.5) == pytest.approx(5.0)


def test_sustained_engagement_counts_seconds_above_the_threshold():
    # baseline = 0th percentile = 0; sd of [0,0,0,0,10] = 4.0; threshold = 4.0
    series = [0.0, 0.0, 0.0, 0.0, 10.0]
    seconds = M.sustained_engagement(
        series, tr_seconds=1.0, baseline_percentile=0.0, threshold_sd=1.0
    )
    assert seconds == pytest.approx(1.0)


def test_sustained_engagement_rejects_an_out_of_range_percentile():
    with pytest.raises(ValueError, match="baseline_percentile"):
        M.sustained_engagement([1.0, 2.0], 1.0, baseline_percentile=150.0, threshold_sd=1.0)


def test_spatial_dispersion_is_the_population_sd_over_parcels():
    assert M.spatial_dispersion([1.0, 1.0, 3.0, 3.0]) == pytest.approx(1.0)


def test_spatial_dispersion_needs_at_least_two_parcels():
    with pytest.raises(ValueError, match="at least 2 parcels"):
        M.spatial_dispersion([1.0])


def test_spatial_entropy_of_a_uniform_profile_is_log_p():
    assert M.spatial_entropy([1.0, 1.0, 1.0, 1.0]) == pytest.approx(np.log(4))


def test_spatial_entropy_of_a_concentrated_profile_is_zero():
    assert M.spatial_entropy([0.0, 0.0, 5.0, 0.0]) == pytest.approx(0.0)


def test_spatial_entropy_raises_on_an_all_zero_vector():
    with pytest.raises(ValueError, match="all-zero"):
        M.spatial_entropy([0.0, 0.0, 0.0])


def test_spatial_entropy_raises_on_an_all_negative_vector():
    with pytest.raises(ValueError, match="non-negative"):
        M.spatial_entropy([-1.0, -2.0, -3.0])


def test_spatial_entropy_raises_on_a_mixed_sign_vector():
    with pytest.raises(ValueError, match="non-negative"):
        M.spatial_entropy([1.0, -2.0, 3.0])


def test_network_integration_is_one_when_every_parcel_moves_together():
    base = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    matrix = np.vstack([base, base * 2, base * 3, base * 4])
    value = M.network_integration(matrix, ["A", "A", "B", "B"])
    assert value == pytest.approx(1.0)


def test_network_integration_is_below_one_when_networks_are_segregated():
    within_a = np.array([1.0, 2.0, 3.0, 4.0])
    within_b = np.array([4.0, 1.0, 3.0, 2.0])
    matrix = np.vstack([within_a, within_a * 1.5, within_b, within_b * 1.5])
    assert M.network_integration(matrix, ["A", "A", "B", "B"]) < 1.0


def test_network_integration_needs_two_networks():
    matrix = np.random.default_rng(0).normal(size=(4, 10))
    with pytest.raises(ValueError, match=">= 2 distinct networks"):
        M.network_integration(matrix, ["A", "A", "A", "A"])


def test_network_integration_rejects_a_constant_parcel():
    matrix = np.vstack([np.ones(5), np.arange(5.0), np.arange(5.0) * 2, np.arange(5.0) * 3])
    with pytest.raises(ValueError, match="constant time series"):
        M.network_integration(matrix, ["A", "A", "B", "B"])


def test_representational_differentiation_needs_two_concepts():
    with pytest.raises(ValueError, match=">= 2 concepts"):
        M.representational_differentiation({"break_even": np.array([1.0, 2.0, 3.0])})


def test_representational_differentiation_of_anticorrelated_concepts_is_two():
    patterns = {"a": np.array([1.0, 2.0, 3.0, 4.0]), "b": np.array([4.0, 3.0, 2.0, 1.0])}
    assert M.representational_differentiation(patterns) == pytest.approx(2.0)


def test_within_concept_consistency_needs_two_variants():
    with pytest.raises(ValueError, match=">= 2 variants"):
        M.within_concept_consistency({"v1": np.array([1.0, 2.0, 3.0])})


def test_within_concept_consistency_of_identical_variants_is_one():
    pattern = np.array([1.0, 5.0, 2.0, 4.0])
    value = M.within_concept_consistency({"v1": pattern, "v2": pattern * 3})
    assert value == pytest.approx(1.0)


def test_window_slice_rejects_an_empty_window():
    with pytest.raises(ValueError, match="selects no samples"):
        M.window_slice(10, tr_seconds=1.0, window=(50.0, 60.0))


def test_window_slice_full_series_when_window_is_none():
    assert M.window_slice(10, 1.0, None) == slice(0, 10)


# -- driver ------------------------------------------------------------------


def _matrix(seed=0, n_parcels=4, n_timesteps=20):
    rng = np.random.default_rng(seed)
    time = np.linspace(0, 3 * np.pi, n_timesteps)
    return np.vstack(
        [np.sin(time + i) * (i + 1) + rng.normal(0, 0.05, n_timesteps) for i in range(n_parcels)]
    )


def test_compute_all_metrics_shape_and_keys(metrics_config):
    parcel_df = make_parcel_frame("s1", _matrix(), NETWORKS)
    metrics = M.compute_all_metrics(parcel_df, metrics_config)

    per_parcel = metrics[metrics["parcel_id"] != M.WHOLE_CORTEX]
    whole = metrics[metrics["parcel_id"] == M.WHOLE_CORTEX]

    assert set(per_parcel["metric"]) == set(M.TIMESERIES_METRICS)
    assert len(per_parcel) == 4 * len(M.TIMESERIES_METRICS)
    assert set(whole["metric"]) == set(M.SPATIAL_METRICS)
    assert set(whole["network"]) == {"__all__"}


def test_compute_all_metrics_records_every_free_parameter(metrics_config):
    parcel_df = make_parcel_frame("s1", _matrix(), NETWORKS)
    metrics = M.compute_all_metrics(parcel_df, metrics_config)
    for column in (
        "smoothing_kernel", "smoothing_width_s", "tr_seconds", "instructional_window_s",
        "baseline_percentile", "sustained_threshold_sd", "spatial_basis",
    ):
        assert metrics[column].notna().all()
    assert set(metrics["spatial_basis"]) == {M.SPATIAL_BASIS}


def test_compute_all_metrics_rejects_a_ragged_table(metrics_config):
    """A parcel missing one timepoint must stop the run, not shorten a series."""
    parcel_df = make_parcel_frame("s1", _matrix(), NETWORKS)
    with pytest.raises(ValueError, match="ragged"):
        M.compute_all_metrics(parcel_df.iloc[:-1], metrics_config)


def test_compute_all_metrics_on_an_empty_table_raises(metrics_config):
    import pandas as pd

    empty = pd.DataFrame(
        columns=["stimulus_id", "time_index", "parcel_id", "network", "mean_bold"]
    )
    with pytest.raises(ValueError, match="empty"):
        M.compute_all_metrics(empty, metrics_config)
