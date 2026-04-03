from __future__ import annotations

"""
Backward-compatible access to dataframe helpers and batch operators.

The dataframe-first batch surface now lives in `osmh3nx.batch`, but existing
imports from `osmh3nx.data` are kept working here on purpose.
"""

from .batch import (
    BatchDriveshedResult,
    BatchNearestTargetResult,
    BatchODRouteResult,
    H3BatchGraphContext,
    build_calibrated_h3_graph_for_points,
    build_od_points_gdf,
    build_points_search_polygon,
    ensure_point_gdf,
    load_od_pairs,
    prepare_od_dataframe,
    required_od_columns,
    run_batch_drivesheds,
    run_batch_nearest_target_assignment,
    run_batch_od_routes,
)

__all__ = [
    "BatchDriveshedResult",
    "BatchNearestTargetResult",
    "BatchODRouteResult",
    "H3BatchGraphContext",
    "build_calibrated_h3_graph_for_points",
    "build_od_points_gdf",
    "build_points_search_polygon",
    "ensure_point_gdf",
    "load_od_pairs",
    "prepare_od_dataframe",
    "required_od_columns",
    "run_batch_drivesheds",
    "run_batch_nearest_target_assignment",
    "run_batch_od_routes",
]
