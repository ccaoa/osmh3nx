from __future__ import annotations

import math

import h3
import networkx as nx
import pytest
from shapely.geometry import Point

from osmh3nx import driveshed as dshed


def test_estimate_driveshed_buffer_miles_matches_45km_default() -> None:
    buffer_miles = dshed.estimate_driveshed_buffer_miles(
        max_travel_minutes=20.0,
        buffer_speed_mph=60.0,
        buffer_factor=dshed.DEFAULT_SEARCH_BUFFER_FACTOR,
        min_buffer_miles=0.0,
    )

    assert math.isclose(buffer_miles, 27.96170365068003, rel_tol=0.0, abs_tol=1e-9)


def test_estimate_driveshed_buffer_miles_respects_minimum() -> None:
    buffer_miles = dshed.estimate_driveshed_buffer_miles(
        max_travel_minutes=1.0,
        buffer_speed_mph=10.0,
        buffer_factor=1.0,
        min_buffer_miles=2.0,
    )

    assert buffer_miles == 2.0


def test_build_driveshed_search_polygon_contains_origin() -> None:
    origin = Point(-79.94143, 37.27097)
    polygon = dshed.build_driveshed_search_polygon(
        origin,
        max_travel_minutes=5.0,
        min_buffer_miles=0.0,
    )

    assert polygon.contains(origin) or polygon.touches(origin)


def test_build_h3_driveshed_from_prebuilt_graph(
    sample_h3_cells: dict[str, str],
) -> None:
    origin_cell = sample_h3_cells["origin_cell"]
    neighbor_cell = sample_h3_cells["neighbor_cell"]
    lat, lng = h3.cell_to_latlng(origin_cell)
    origin = Point(lng, lat)

    graph = nx.Graph()
    graph.add_edge(
        origin_cell,
        neighbor_cell,
        travel_time_route=120.0,
        observed_step_time_raw_sec=120.0,
        step_time_floored_sec=120.0,
        step_time_route_sec=120.0,
        step_time_postcalibrated_sec=120.0,
        centroid_dist_miles=0.5,
        floor_applied=False,
    )

    result = dshed.build_h3_driveshed_from_point(
        origin,
        max_travel_minutes=5.0,
        h3_res=10,
        weight_attr="travel_time_route",
        h3_graph=graph,
        search_polygon_wgs84=None,
        snap_max_k=0,
    )

    assert result.origin_h3_cell == origin_cell
    assert result.origin_h3_cell_graph == origin_cell
    assert set(result.reachable_cells_gdf["h3_cell"]) == {origin_cell, neighbor_cell}
    assert result.reachable_edges_gdf["travel_time_sec"].tolist() == [120.0]


def test_build_h3_driveshed_from_prebuilt_graph_rejects_missing_weight(
    sample_h3_cells: dict[str, str],
) -> None:
    origin_cell = sample_h3_cells["origin_cell"]
    neighbor_cell = sample_h3_cells["neighbor_cell"]
    lat, lng = h3.cell_to_latlng(origin_cell)
    origin = Point(lng, lat)

    graph = nx.Graph()
    graph.add_edge(origin_cell, neighbor_cell, some_other_weight=30.0)

    with pytest.raises(ValueError, match="weight_attr 'travel_time_route' not present"):
        dshed.build_h3_driveshed_from_point(
            origin,
            max_travel_minutes=5.0,
            h3_res=10,
            weight_attr="travel_time_route",
            h3_graph=graph,
            snap_max_k=0,
        )
