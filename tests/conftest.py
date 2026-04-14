from __future__ import annotations

from typing import Any

import geopandas as gpd
import h3
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon


def h3_cell_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


@pytest.fixture
def sample_h3_cells() -> dict[str, Any]:
    origin_cell = h3.latlng_to_cell(37.27097, -79.94143, 10)
    neighbor_cell = sorted(h3.grid_ring(origin_cell, 1))[0]
    parent_cell = h3.cell_to_parent(origin_cell, 9)
    return {
        "origin_cell": origin_cell,
        "neighbor_cell": neighbor_cell,
        "parent_cell": parent_cell,
    }


@pytest.fixture
def sample_lookup_df(sample_h3_cells: dict[str, Any]) -> pd.DataFrame:
    origin_cell = sample_h3_cells["origin_cell"]
    neighbor_cell = sample_h3_cells["neighbor_cell"]
    parent_cell = sample_h3_cells["parent_cell"]
    return pd.DataFrame(
        [
            {
                "origin_id": "origin_a",
                "graph_group_id": "all",
                "status": "ok",
                "h3_cell": origin_cell,
                "h3_res": 10,
                "source_h3_res": 10,
                "is_upsampled": False,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 10,
                "origin_h3_cell": origin_cell,
                "origin_h3_cell_graph": origin_cell,
                "travel_time_sec": 0.0,
                "travel_time_minutes": 0.0,
                "path_n_cells": 1,
                "is_origin_cell_graph": True,
                "n_child_cells_from_source": None,
                "max_travel_minutes": 20.0,
            },
            {
                "origin_id": "origin_a",
                "graph_group_id": "all",
                "status": "ok",
                "h3_cell": neighbor_cell,
                "h3_res": 10,
                "source_h3_res": 10,
                "is_upsampled": False,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 10,
                "origin_h3_cell": origin_cell,
                "origin_h3_cell_graph": origin_cell,
                "travel_time_sec": 90.0,
                "travel_time_minutes": 1.5,
                "path_n_cells": 2,
                "is_origin_cell_graph": False,
                "n_child_cells_from_source": None,
                "max_travel_minutes": 20.0,
            },
            {
                "origin_id": "origin_b",
                "graph_group_id": "group_b",
                "status": "ok",
                "h3_cell": neighbor_cell,
                "h3_res": 10,
                "source_h3_res": 10,
                "is_upsampled": False,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 10,
                "origin_h3_cell": neighbor_cell,
                "origin_h3_cell_graph": neighbor_cell,
                "travel_time_sec": 0.0,
                "travel_time_minutes": 0.0,
                "path_n_cells": 1,
                "is_origin_cell_graph": True,
                "n_child_cells_from_source": None,
                "max_travel_minutes": 20.0,
            },
            {
                "origin_id": "origin_a",
                "graph_group_id": "all",
                "status": "ok",
                "h3_cell": parent_cell,
                "h3_res": 9,
                "source_h3_res": 10,
                "is_upsampled": True,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 9,
                "origin_h3_cell": origin_cell,
                "origin_h3_cell_graph": origin_cell,
                "travel_time_sec": 45.0,
                "travel_time_minutes": 0.75,
                "path_n_cells": 2,
                "is_origin_cell_graph": True,
                "n_child_cells_from_source": 2,
                "max_travel_minutes": 20.0,
            },
        ]
    )


@pytest.fixture
def sample_unique_cells_gdf(sample_h3_cells: dict[str, Any]) -> gpd.GeoDataFrame:
    origin_cell = sample_h3_cells["origin_cell"]
    neighbor_cell = sample_h3_cells["neighbor_cell"]
    parent_cell = sample_h3_cells["parent_cell"]
    return gpd.GeoDataFrame(
        [
            {
                "h3_cell": origin_cell,
                "h3_res": 10,
                "source_h3_res": 10,
                "is_upsampled": False,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 10,
                "geometry": h3_cell_polygon(origin_cell),
            },
            {
                "h3_cell": neighbor_cell,
                "h3_res": 10,
                "source_h3_res": 10,
                "is_upsampled": False,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 10,
                "geometry": h3_cell_polygon(neighbor_cell),
            },
            {
                "h3_cell": parent_cell,
                "h3_res": 9,
                "source_h3_res": 10,
                "is_upsampled": True,
                "upsampled_source_h3_res": 10,
                "upsampled_target_h3_res": 9,
                "geometry": h3_cell_polygon(parent_cell),
            },
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )


@pytest.fixture
def sample_points_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b", "c"]},
        geometry=[
            Point(-79.94143, 37.27097),
            Point(-79.94500, 37.27200),
            Point(-79.94300, 37.26850),
        ],
        crs="EPSG:4326",
    )
