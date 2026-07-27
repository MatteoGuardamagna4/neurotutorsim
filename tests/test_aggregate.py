"""Vertex -> parcel -> network aggregation, equations (6) and (7)."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.tribe.aggregate import (
    AtlasError,
    assert_covers_mesh,
    parcels_to_networks,
    vertices_to_parcels,
)

# 2 timesteps x 6 vertices. Parcels are (0,1), (2,3), (4,5).
PREDICTIONS = np.array(
    [
        [1.0, 3.0, 10.0, 20.0, 100.0, 100.0],
        [0.0, 0.0, 2.0, 4.0, 5.0, 15.0],
    ],
    dtype=np.float32,
)


def test_equation_6_is_a_plain_unweighted_vertex_mean(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "equal", stimulus_id="s1")

    at = lambda p, t: float(  # noqa: E731 - table lookup, reads better inline
        parcels[(parcels["parcel_id"] == p) & (parcels["time_index"] == t)]["mean_bold"].iloc[0]
    )
    assert at(1, 0) == pytest.approx(2.0)     # (1 + 3) / 2
    assert at(2, 0) == pytest.approx(15.0)    # (10 + 20) / 2
    assert at(3, 0) == pytest.approx(100.0)   # (100 + 100) / 2
    assert at(1, 1) == pytest.approx(0.0)
    assert at(2, 1) == pytest.approx(3.0)
    assert at(3, 1) == pytest.approx(10.0)


def test_sd_bold_records_the_within_parcel_spread_that_the_mean_discards(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "equal", stimulus_id="s1")
    row = parcels[(parcels["parcel_id"] == 3) & (parcels["time_index"] == 0)]
    assert float(row["sd_bold"].iloc[0]) == pytest.approx(0.0)  # 100, 100
    row = parcels[(parcels["parcel_id"] == 3) & (parcels["time_index"] == 1)]
    assert float(row["sd_bold"].iloc[0]) == pytest.approx(5.0)  # 5, 15 -> ddof=0


def test_parcel_table_shape_and_provenance(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "area", stimulus_id="s1")
    assert len(parcels) == 3 * 2
    assert list(parcels.columns) == [
        "stimulus_id", "time_index", "parcel_id", "network", "mean_bold", "sd_bold"
    ]
    provenance = parcels.attrs["provenance"]
    assert provenance["equation"] == "6"
    assert provenance["vertex_to_parcel"] == "unweighted mean"
    assert provenance["parcel_weighting"] == "area"
    assert provenance["atlas_name"] == "toy3"


def test_equation_7_equal_weighting(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "equal", stimulus_id="s1")
    networks = parcels_to_networks(parcels, "equal", toy_atlas)

    a0 = networks[(networks["network"] == "A") & (networks["time_index"] == 0)]
    assert float(a0["mean_bold"].iloc[0]) == pytest.approx((2.0 + 15.0) / 2)
    b0 = networks[(networks["network"] == "B") & (networks["time_index"] == 0)]
    assert float(b0["mean_bold"].iloc[0]) == pytest.approx(100.0)
    assert set(networks["weighting"]) == {"equal"}


def test_equation_7_area_weighting(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "area", stimulus_id="s1")
    networks = parcels_to_networks(parcels, "area", toy_atlas)

    # network A = parcels 1 (area 2, mean 2.0) and 2 (area 4, mean 15.0)
    expected = (2.0 * 2.0 + 4.0 * 15.0) / (2.0 + 4.0)
    a0 = networks[(networks["network"] == "A") & (networks["time_index"] == 0)]
    assert float(a0["mean_bold"].iloc[0]) == pytest.approx(expected)
    assert set(networks["weighting"]) == {"area"}


def test_the_two_weightings_disagree_so_the_choice_is_recorded_not_cosmetic(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "area", stimulus_id="s1")
    equal = parcels_to_networks(parcels, "equal", toy_atlas)
    area = parcels_to_networks(parcels, "area", toy_atlas)
    merged = equal.merge(area, on=["stimulus_id", "time_index", "network"], suffixes=("_e", "_a"))
    assert not np.allclose(merged["mean_bold_e"], merged["mean_bold_a"])


def test_atlas_that_does_not_cover_the_mesh_raises(toy_atlas):
    short = np.zeros((2, 5), dtype=np.float32)
    with pytest.raises(AtlasError, match="must cover exactly the mesh"):
        vertices_to_parcels(short, toy_atlas, "equal", stimulus_id="s1")


def test_assert_covers_mesh_is_exact(toy_atlas):
    assert_covers_mesh(toy_atlas, 6)
    with pytest.raises(AtlasError):
        assert_covers_mesh(toy_atlas, 7)


def test_unknown_weighting_raises(toy_atlas):
    with pytest.raises(ValueError, match="weighting must be one of"):
        vertices_to_parcels(PREDICTIONS, toy_atlas, "inverse_variance", stimulus_id="s1")


def test_one_dimensional_prediction_raises(toy_atlas):
    with pytest.raises(ValueError, match="2-D"):
        vertices_to_parcels(np.zeros(6, dtype=np.float32), toy_atlas, "equal", stimulus_id="s1")


def test_network_weighting_rejects_an_unknown_parcel(toy_atlas):
    parcels = vertices_to_parcels(PREDICTIONS, toy_atlas, "area", stimulus_id="s1")
    parcels.loc[0, "parcel_id"] = 99
    with pytest.raises(AtlasError, match="does not define"):
        parcels_to_networks(parcels, "area", toy_atlas)


def test_labels_length_is_the_single_assignment_guarantee(toy_atlas):
    """One label slot per vertex means a vertex cannot map to two parcels."""
    assert toy_atlas.labels.shape[0] == toy_atlas.n_vertices
    assert toy_atlas.labels.ndim == 1
    doubled = replace(toy_atlas, labels=np.tile(toy_atlas.labels, 2))
    with pytest.raises(AtlasError):
        vertices_to_parcels(PREDICTIONS, doubled, "equal", stimulus_id="s1")
