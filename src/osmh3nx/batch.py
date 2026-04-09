from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import h3
import networkx as nx
import pandas as pd
from shapely import wkt as shapely_wkt
from shapely.geometry import Point, Polygon, box

from . import calibrate as calx
from . import driveshed as dshed
from . import network_h3 as hnetx
from . import network_osm as onetx
from . import spatial as spatx

DEFAULT_POINT_CRS: str = "EPSG:4326"
DEFAULT_GRAPH_BUFFER_MILES: float = 10.0
DEFAULT_MIN_SQUARE_SIDE_M: float = 1000.0


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now_text()}] {message}", flush=True)


@dataclass
class H3BatchGraphContext:
    graph_group_id: str
    group_key: Tuple[Any, ...]
    group_columns: Tuple[str, ...]
    search_polygon_wgs84: Polygon
    osm_graph: nx.MultiDiGraph
    h3_graph: nx.Graph
    calibration_profile: calx.CalibrationProfile
    h3_res: int
    weight_attr: str
    report_weight_attr: str


@dataclass
class BatchODRouteResult:
    od_results_df: pd.DataFrame
    routes_gdf: gpd.GeoDataFrame
    route_hexes_gdf: gpd.GeoDataFrame
    search_polygons_gdf: gpd.GeoDataFrame
    graph_contexts: Dict[str, H3BatchGraphContext]


@dataclass
class BatchNearestTargetResult:
    assigned_sources_gdf: gpd.GeoDataFrame
    routes_gdf: gpd.GeoDataFrame
    route_hexes_gdf: gpd.GeoDataFrame
    search_polygons_gdf: gpd.GeoDataFrame
    graph_contexts: Dict[str, H3BatchGraphContext]


@dataclass
class BatchDriveshedResult:
    origins_gdf: gpd.GeoDataFrame
    search_polygons_gdf: gpd.GeoDataFrame
    driveshed_polygons_gdf: gpd.GeoDataFrame
    driveshed_cells_gdf: gpd.GeoDataFrame
    driveshed_edges_gdf: gpd.GeoDataFrame
    driveshed_cells_upsampled_gdf: gpd.GeoDataFrame
    driveshed_polygons_upsampled_gdf: gpd.GeoDataFrame
    graph_contexts: Dict[str, H3BatchGraphContext]


def required_od_columns() -> List[str]:
    return ["count", "city", "state", "category", "origin", "destination"]


def load_od_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in required_od_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration CSV: {missing}")

    out = df.copy()
    out["pair_id"] = out["count"].astype(int)
    out["origin_geom"] = out["origin"].apply(_coerce_point_value)
    out["destination_geom"] = out["destination"].apply(_coerce_point_value)
    out = out.sort_values("pair_id").reset_index(drop=True)
    return out


def build_od_points_gdf(od_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in od_pairs.iterrows():
        base = {
            "pair_id": int(row["pair_id"]),
            "count": int(row["count"]),
            "city": row["city"],
            "state": row["state"],
            "category": row["category"],
            "origin": row["origin"],
            "destination": row["destination"],
        }
        rows.append(
            {
                **base,
                "point_role": "origin",
                "point_wkt": row["origin"],
                "geometry": row["origin_geom"],
            }
        )
        rows.append(
            {
                **base,
                "point_role": "destination",
                "point_wkt": row["destination"],
                "geometry": row["destination_geom"],
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=DEFAULT_POINT_CRS)


def ensure_point_gdf(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    lon_col: str = "lon",
    lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    target_crs: str = DEFAULT_POINT_CRS,
) -> gpd.GeoDataFrame:
    if isinstance(data, gpd.GeoDataFrame):
        out = data.copy()
        if geometry_col in out.columns and geometry_col != out.geometry.name:
            out = out.set_geometry(geometry_col)
        if out.crs is None:
            out = out.set_crs(input_crs)
    else:
        out = data.copy()
        if geometry_col in out.columns:
            geom_series = out[geometry_col].apply(_coerce_point_value)
            out = gpd.GeoDataFrame(out, geometry=geom_series, crs=input_crs)
        else:
            missing = [col for col in (lon_col, lat_col) if col not in out.columns]
            if missing:
                raise ValueError(
                    f"Could not find point geometry columns. Missing: {missing}. "
                    f"Expected geometry_col='{geometry_col}' or lon/lat columns."
                )
            out = gpd.GeoDataFrame(
                out.copy(),
                geometry=gpd.points_from_xy(out[lon_col], out[lat_col]),
                crs=input_crs,
            )

    out = _ensure_point_geometry_series(out)
    if target_crs and str(out.crs).lower() != str(target_crs).lower():
        out = out.to_crs(target_crs)
    return out


def prepare_od_dataframe(
    data: pd.DataFrame | gpd.GeoDataFrame,
    *,
    origin_geometry_col: Optional[str] = "origin_geometry",
    destination_geometry_col: Optional[str] = "destination_geometry",
    origin_lon_col: str = "origin_lon",
    origin_lat_col: str = "origin_lat",
    destination_lon_col: str = "destination_lon",
    destination_lat_col: str = "destination_lat",
    input_crs: str = DEFAULT_POINT_CRS,
    target_crs: str = DEFAULT_POINT_CRS,
) -> pd.DataFrame:
    out = pd.DataFrame(data.copy())
    source_gdf = data if isinstance(data, gpd.GeoDataFrame) else None

    out["origin_geom"] = _resolve_point_series(
        out,
        geometry_col=origin_geometry_col,
        lon_col=origin_lon_col,
        lat_col=origin_lat_col,
        input_crs=input_crs,
        target_crs=target_crs,
        source_gdf=source_gdf,
        allow_active_geometry=True,
    )
    out["destination_geom"] = _resolve_point_series(
        out,
        geometry_col=destination_geometry_col,
        lon_col=destination_lon_col,
        lat_col=destination_lat_col,
        input_crs=input_crs,
        target_crs=target_crs,
        source_gdf=source_gdf,
        allow_active_geometry=False,
    )
    return out


def build_points_search_polygon(
    points: pd.DataFrame | gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    lon_col: str = "lon",
    lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    buffer_miles: float = DEFAULT_GRAPH_BUFFER_MILES,
    min_square_side_m: float = DEFAULT_MIN_SQUARE_SIDE_M,
) -> Polygon:
    points_gdf = ensure_point_gdf(
        points,
        geometry_col=geometry_col,
        lon_col=lon_col,
        lat_col=lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    if points_gdf.empty:
        raise ValueError("points is empty.")
    if buffer_miles <= 0:
        raise ValueError("buffer_miles must be > 0.")
    if min_square_side_m <= 0:
        raise ValueError("min_square_side_m must be > 0.")

    proj_crs = points_gdf.estimate_utm_crs() or "EPSG:3857"
    points_proj = points_gdf.to_crs(proj_crs)
    minx, miny, maxx, maxy = [float(v) for v in points_proj.total_bounds]
    side = max(maxx - minx, maxy - miny, float(min_square_side_m))
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    square_proj = box(
        cx - side / 2.0, cy - side / 2.0, cx + side / 2.0, cy + side / 2.0
    )
    buffered_proj = square_proj.buffer(onetx.miles_to_meters(buffer_miles))
    return (
        gpd.GeoSeries([buffered_proj], crs=proj_crs).to_crs(DEFAULT_POINT_CRS).iloc[0]
    )


def build_points_convex_hull_search_polygon(
    points: pd.DataFrame | gpd.GeoDataFrame,
    *,
    geometry_col: str = "geometry",
    lon_col: str = "lon",
    lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    buffer_miles: float = spatx.DEFAULT_BUFFERED_CONVEX_HULL_BUFFER_MILES,
) -> Polygon:
    points_gdf = ensure_point_gdf(
        points,
        geometry_col=geometry_col,
        lon_col=lon_col,
        lat_col=lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    if points_gdf.empty:
        raise ValueError("points is empty.")

    hull_gdf = spatx.create_buffered_convex_hull(
        points_gdf,
        buffer_miles=buffer_miles,
    )
    return hull_gdf.geometry.iloc[0]


def build_calibrated_h3_graph_for_points(
    points: pd.DataFrame | gpd.GeoDataFrame,
    *,
    graph_group_id: str = "all",
    group_key: Tuple[Any, ...] = (),
    group_columns: Sequence[str] = (),
    geometry_col: str = "geometry",
    lon_col: str = "lon",
    lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    search_polygon_wgs84: Optional[Polygon] = None,
    buffer_miles: float = DEFAULT_GRAPH_BUFFER_MILES,
    min_square_side_m: float = DEFAULT_MIN_SQUARE_SIDE_M,
    h3_res: Optional[int] = None,
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME,
    calibration_profile_overrides: Optional[Mapping[str, Any]] = None,
    weight_attr: Optional[str] = None,
    report_weight_attr: Optional[str] = None,
    osm_cache_dir: Optional[str] = onetx.DEFAULT_CACHE_DIR,
    osm_force_refresh: bool = False,
    G_osm: Optional[nx.MultiDiGraph] = None,
    h3_graph: Optional[nx.Graph] = None,
) -> H3BatchGraphContext:
    profile = calx.get_calibration_profile(
        calibration_profile_name,
        overrides=calibration_profile_overrides,
    )
    resolved_h3_res = int(
        h3_res if h3_res is not None else calx.get_default_h3_res(profile)
    )
    resolved_weight_attr = str(
        weight_attr or calx.get_default_query_weight_attr(profile)
    )
    resolved_report_weight_attr = str(
        report_weight_attr or calx.get_default_report_weight_attr(profile)
    )

    if h3_graph is None:
        if search_polygon_wgs84 is None:
            search_polygon_wgs84 = build_points_search_polygon(
                points,
                geometry_col=geometry_col,
                lon_col=lon_col,
                lat_col=lat_col,
                input_crs=input_crs,
                buffer_miles=buffer_miles,
                min_square_side_m=min_square_side_m,
            )
        if G_osm is None:
            G_osm = onetx.download_osm_drive_graph_for_polygon(
                search_polygon_wgs84,
                cache_dir=osm_cache_dir,
                force_refresh=osm_force_refresh,
            )
        h3_graph, profile = calx.build_calibrated_h3_graph_from_osm(
            G_osm,
            h3_res=resolved_h3_res,
            profile_name=calibration_profile_name,
            profile_overrides=calibration_profile_overrides,
        )
    else:
        if search_polygon_wgs84 is None:
            search_polygon_wgs84 = build_points_search_polygon(
                points,
                geometry_col=geometry_col,
                lon_col=lon_col,
                lat_col=lat_col,
                input_crs=input_crs,
                buffer_miles=buffer_miles,
                min_square_side_m=min_square_side_m,
            )
        if G_osm is None:
            G_osm = nx.MultiDiGraph()
        graph_profile_name = h3_graph.graph.get("calibration_profile_name")
        if graph_profile_name:
            profile = calx.get_calibration_profile(str(graph_profile_name))
        graph_h3_res = h3_graph.graph.get("h3_res")
        if graph_h3_res is not None and int(graph_h3_res) != resolved_h3_res:
            raise ValueError(
                f"Provided h3_graph has h3_res={graph_h3_res}, but h3_res={resolved_h3_res} was requested."
            )
        resolved_weight_attr = str(
            weight_attr
            or h3_graph.graph.get("default_query_weight_attr")
            or calx.get_default_query_weight_attr(profile)
        )
        resolved_report_weight_attr = str(
            report_weight_attr
            or h3_graph.graph.get("report_weight_attr")
            or calx.get_default_report_weight_attr(profile)
        )

    return H3BatchGraphContext(
        graph_group_id=str(graph_group_id),
        group_key=tuple(group_key),
        group_columns=tuple(group_columns),
        search_polygon_wgs84=search_polygon_wgs84,
        osm_graph=G_osm,
        h3_graph=h3_graph,
        calibration_profile=profile,
        h3_res=resolved_h3_res,
        weight_attr=resolved_weight_attr,
        report_weight_attr=resolved_report_weight_attr,
    )


def run_batch_od_routes(
    od_pairs: pd.DataFrame | gpd.GeoDataFrame,
    *,
    pair_id_col: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
    origin_geometry_col: Optional[str] = "origin_geometry",
    destination_geometry_col: Optional[str] = "destination_geometry",
    origin_lon_col: str = "origin_lon",
    origin_lat_col: str = "origin_lat",
    destination_lon_col: str = "destination_lon",
    destination_lat_col: str = "destination_lat",
    input_crs: str = DEFAULT_POINT_CRS,
    buffer_miles: float = DEFAULT_GRAPH_BUFFER_MILES,
    min_square_side_m: float = DEFAULT_MIN_SQUARE_SIDE_M,
    h3_res: Optional[int] = None,
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME,
    calibration_profile_overrides: Optional[Mapping[str, Any]] = None,
    weight_attr: Optional[str] = None,
    report_weight_attr: Optional[str] = None,
    snap_max_k: int = 10,
    osm_cache_dir: Optional[str] = onetx.DEFAULT_CACHE_DIR,
    osm_force_refresh: bool = False,
    continue_on_error: bool = True,
) -> BatchODRouteResult:
    od_df = prepare_od_dataframe(
        od_pairs,
        origin_geometry_col=origin_geometry_col,
        destination_geometry_col=destination_geometry_col,
        origin_lon_col=origin_lon_col,
        origin_lat_col=origin_lat_col,
        destination_lon_col=destination_lon_col,
        destination_lat_col=destination_lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    od_df, resolved_pair_id_col = _ensure_id_column(od_df, pair_id_col, "pair")

    result_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    route_hex_rows: List[Dict[str, Any]] = []
    search_rows: List[Dict[str, Any]] = []
    graph_contexts: Dict[str, H3BatchGraphContext] = {}

    for graph_group_id, group_key, group_frame in _iter_groups(od_df, group_cols):
        points_gdf = gpd.GeoDataFrame(
            pd.concat(
                [
                    gpd.GeoDataFrame(
                        group_frame.copy(),
                        geometry="origin_geom",
                        crs=DEFAULT_POINT_CRS,
                    )[["origin_geom"]].rename(columns={"origin_geom": "geometry"}),
                    gpd.GeoDataFrame(
                        group_frame.copy(),
                        geometry="destination_geom",
                        crs=DEFAULT_POINT_CRS,
                    )[["destination_geom"]].rename(
                        columns={"destination_geom": "geometry"}
                    ),
                ],
                ignore_index=True,
            ),
            geometry="geometry",
            crs=DEFAULT_POINT_CRS,
        )
        context = build_calibrated_h3_graph_for_points(
            points_gdf,
            graph_group_id=graph_group_id,
            group_key=group_key,
            group_columns=tuple(group_cols or ()),
            buffer_miles=buffer_miles,
            min_square_side_m=min_square_side_m,
            h3_res=h3_res,
            calibration_profile_name=calibration_profile_name,
            calibration_profile_overrides=calibration_profile_overrides,
            weight_attr=weight_attr,
            report_weight_attr=report_weight_attr,
            osm_cache_dir=osm_cache_dir,
            osm_force_refresh=osm_force_refresh,
        )
        graph_contexts[graph_group_id] = context
        search_rows.append(
            _build_search_polygon_row(
                graph_group_id=graph_group_id,
                group_columns=tuple(group_cols or ()),
                group_key=group_key,
                context=context,
            )
        )

        for _, row in group_frame.iterrows():
            base_row = dict(row)
            base_row["graph_group_id"] = graph_group_id
            base_row["h3_res"] = int(context.h3_res)
            base_row["weight_attr"] = context.weight_attr
            base_row["report_weight_attr"] = context.report_weight_attr
            base_row["calibration_profile_name"] = context.calibration_profile.name

            try:
                route_result = _solve_h3_route(
                    h3_graph=context.h3_graph,
                    origin_wgs84=row["origin_geom"],
                    destination_wgs84=row["destination_geom"],
                    h3_res=context.h3_res,
                    weight_attr=context.weight_attr,
                    report_weight_attr=context.report_weight_attr,
                    snap_max_k=snap_max_k,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                route_result = {
                    "status": f"error:{type(exc).__name__}",
                    "error_message": str(exc),
                    "path_cells": None,
                    "travel_time_sec": None,
                    "travel_time_postcalibrated_sec": None,
                    "travel_miles": None,
                    "origin_cell": None,
                    "destination_cell": None,
                    "origin_cell_graph": None,
                    "destination_cell_graph": None,
                    "geometry": None,
                }

            out_row = {
                **base_row,
                "status": route_result["status"],
                "error_message": route_result.get("error_message"),
                "origin_h3_cell": route_result.get("origin_cell"),
                "destination_h3_cell": route_result.get("destination_cell"),
                "origin_h3_cell_graph": route_result.get("origin_cell_graph"),
                "destination_h3_cell_graph": route_result.get("destination_cell_graph"),
                "h3_path": (
                    "|".join(route_result["path_cells"])
                    if route_result.get("path_cells")
                    else None
                ),
                "h3_path_n_cells": (
                    len(route_result["path_cells"])
                    if route_result.get("path_cells")
                    else 0
                ),
                "h3_travel_time": route_result.get("travel_time_sec"),
                "h3_travel_time_postcalibrated": route_result.get(
                    "travel_time_postcalibrated_sec"
                ),
                "h3_travel_miles": route_result.get("travel_miles"),
            }
            result_rows.append(out_row)

            if (
                route_result.get("path_cells")
                and route_result.get("geometry") is not None
            ):
                route_rows.append(
                    {
                        resolved_pair_id_col: row[resolved_pair_id_col],
                        "graph_group_id": graph_group_id,
                        "status": route_result["status"],
                        "h3_travel_time": route_result.get("travel_time_sec"),
                        "h3_travel_time_postcalibrated": route_result.get(
                            "travel_time_postcalibrated_sec"
                        ),
                        "h3_travel_miles": route_result.get("travel_miles"),
                        "n_cells": len(route_result["path_cells"]),
                        "geometry": route_result["geometry"],
                    }
                )
                route_hex_rows.extend(
                    _build_route_hex_rows(
                        route_id=row[resolved_pair_id_col],
                        path_cells=route_result["path_cells"],
                        graph_group_id=graph_group_id,
                        extra_attrs={
                            resolved_pair_id_col: row[resolved_pair_id_col],
                            "status": route_result["status"],
                        },
                    )
                )

    return BatchODRouteResult(
        od_results_df=pd.DataFrame(result_rows),
        routes_gdf=_rows_to_gdf(route_rows, geometry_col="geometry"),
        route_hexes_gdf=_rows_to_gdf(route_hex_rows, geometry_col="geometry"),
        search_polygons_gdf=_rows_to_gdf(search_rows, geometry_col="geometry"),
        graph_contexts=graph_contexts,
    )


def run_batch_nearest_target_assignment(
    source_points: pd.DataFrame | gpd.GeoDataFrame,
    target_points: pd.DataFrame | gpd.GeoDataFrame,
    *,
    target_id_col: str,
    source_id_col: Optional[str] = None,
    source_geometry_col: str = "geometry",
    target_geometry_col: str = "geometry",
    source_lon_col: str = "lon",
    source_lat_col: str = "lat",
    target_lon_col: str = "lon",
    target_lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    group_cols: Optional[Sequence[str]] = None,
    buffer_miles: float = DEFAULT_GRAPH_BUFFER_MILES,
    min_square_side_m: float = DEFAULT_MIN_SQUARE_SIDE_M,
    h3_res: Optional[int] = None,
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME,
    calibration_profile_overrides: Optional[Mapping[str, Any]] = None,
    weight_attr: Optional[str] = None,
    out_col: str = "nearest_target_id",
    out_path_col: str = "h3_path",
    out_time_col: str = "h3_travel_time",
    out_distance_col: str = "h3_travel_miles",
    tie_break_project_crs: str = "EPSG:3857",
    snap_max_k: int = 10,
    osm_cache_dir: Optional[str] = onetx.DEFAULT_CACHE_DIR,
    osm_force_refresh: bool = False,
    continue_on_error: bool = True,
) -> BatchNearestTargetResult:
    src = ensure_point_gdf(
        source_points,
        geometry_col=source_geometry_col,
        lon_col=source_lon_col,
        lat_col=source_lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    tgt = ensure_point_gdf(
        target_points,
        geometry_col=target_geometry_col,
        lon_col=target_lon_col,
        lat_col=target_lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    src, resolved_source_id_col = _ensure_id_column(src, source_id_col, "source")

    assigned_groups: List[gpd.GeoDataFrame] = []
    route_groups: List[gpd.GeoDataFrame] = []
    route_hex_rows: List[Dict[str, Any]] = []
    search_rows: List[Dict[str, Any]] = []
    graph_contexts: Dict[str, H3BatchGraphContext] = {}

    for graph_group_id, group_key, src_group in _iter_groups(src, group_cols):
        tgt_group = _subset_by_group_key(tgt, group_cols, group_key)
        if tgt_group.empty:
            missing_group = src_group.copy()
            missing_group[out_col] = None
            missing_group["assignment_status"] = "missing_target_group"
            missing_group["graph_group_id"] = graph_group_id
            missing_group["error_message"] = None
            assigned_groups.append(missing_group)
            continue

        combined_points = gpd.GeoDataFrame(
            pd.concat(
                [src_group[["geometry"]], tgt_group[["geometry"]]], ignore_index=True
            ),
            geometry="geometry",
            crs=DEFAULT_POINT_CRS,
        )
        context = build_calibrated_h3_graph_for_points(
            combined_points,
            graph_group_id=graph_group_id,
            group_key=group_key,
            group_columns=tuple(group_cols or ()),
            buffer_miles=buffer_miles,
            min_square_side_m=min_square_side_m,
            h3_res=h3_res,
            calibration_profile_name=calibration_profile_name,
            calibration_profile_overrides=calibration_profile_overrides,
            weight_attr=weight_attr,
            osm_cache_dir=osm_cache_dir,
            osm_force_refresh=osm_force_refresh,
        )
        graph_contexts[graph_group_id] = context
        search_rows.append(
            _build_search_polygon_row(
                graph_group_id=graph_group_id,
                group_columns=tuple(group_cols or ()),
                group_key=group_key,
                context=context,
            )
        )

        try:
            assigned = hnetx.assign_nearest_target_by_h3_network(
                src_group,
                tgt_group,
                target_id_col=target_id_col,
                out_col=out_col,
                h3_graph=context.h3_graph,
                h3_res=context.h3_res,
                weight_attr=context.weight_attr,
                tie_break_project_crs=tie_break_project_crs,
                out_path_col=out_path_col,
                out_time_col=out_time_col,
                out_distance_col=out_distance_col,
                debug_route_breakdown_source_idx=None,
                debug_route_breakdown_max_steps=None,
                snap_max_k=snap_max_k,
            )
        except Exception as exc:
            if not continue_on_error:
                raise
            assigned = src_group.copy()
            assigned[out_col] = None
            assigned[out_path_col] = None
            assigned[out_time_col] = None
            assigned[out_distance_col] = None
            assigned["assignment_status"] = f"error:{type(exc).__name__}"
            assigned["error_message"] = str(exc)
        else:
            assigned["assignment_status"] = assigned[out_col].apply(
                lambda value: "ok" if pd.notna(value) else "unassigned"
            )
            assigned["error_message"] = None

        assigned["graph_group_id"] = graph_group_id
        assigned["h3_res"] = int(context.h3_res)
        assigned["weight_attr"] = context.weight_attr
        assigned["report_weight_attr"] = context.report_weight_attr
        assigned["calibration_profile_name"] = context.calibration_profile.name
        assigned_groups.append(assigned)

        route_gdf = hnetx.build_route_gdf_from_assignment(
            assigned,
            src_id_col=resolved_source_id_col,
            path_col=out_path_col,
            time_col=out_time_col,
            distance_col=out_distance_col,
            target_id_col=out_col,
        )
        if not route_gdf.empty:
            route_gdf["graph_group_id"] = graph_group_id
            route_gdf["h3_res"] = int(context.h3_res)
            route_gdf["weight_attr"] = context.weight_attr
            route_gdf["calibration_profile_name"] = context.calibration_profile.name
            route_groups.append(route_gdf)

        route_hex_rows.extend(
            _build_route_hex_rows_from_gdf(
                assigned,
                route_id_col=resolved_source_id_col,
                path_col=out_path_col,
                graph_group_id=graph_group_id,
                extra_cols=[out_col, "assignment_status"],
            )
        )

    return BatchNearestTargetResult(
        assigned_sources_gdf=_concat_gdfs(assigned_groups),
        routes_gdf=_concat_gdfs(route_groups),
        route_hexes_gdf=_rows_to_gdf(route_hex_rows, geometry_col="geometry"),
        search_polygons_gdf=_rows_to_gdf(search_rows, geometry_col="geometry"),
        graph_contexts=graph_contexts,
    )


def run_batch_drivesheds(
    origins: pd.DataFrame | gpd.GeoDataFrame,
    *,
    origin_id_col: Optional[str] = None,
    graph_group_cols: Optional[Sequence[str]] = None,
    share_graph_for_all_origins: bool = False,
    geometry_col: str = "geometry",
    lon_col: str = "lon",
    lat_col: str = "lat",
    input_crs: str = DEFAULT_POINT_CRS,
    max_travel_minutes: float = 15.0,
    h3_res: Optional[int] = None,
    weight_attr: Optional[str] = None,
    calibration_profile_name: str = dshed.DEFAULT_CALIBRATION_PROFILE_NAME,
    calibration_profile_overrides: Optional[Dict[str, Any]] = None,
    osm_cache_dir: Optional[str] = dshed.DEFAULT_OSM_CACHE_DIR,
    osm_force_refresh: bool = False,
    search_buffer_speed_mph: float = 60.0,
    search_buffer_factor: float = dshed.DEFAULT_SEARCH_BUFFER_FACTOR,
    search_min_buffer_miles: float = 2.0,
    shared_graph_buffer_miles: float = dshed.DEFAULT_SHARED_GRAPH_BUFFER_MILES,
    snap_max_k: int = 10,
    upsampled_target_resolutions: Optional[Sequence[int]] = None,
    continue_on_error: bool = True,
) -> BatchDriveshedResult:
    origin_gdf = ensure_point_gdf(
        origins,
        geometry_col=geometry_col,
        lon_col=lon_col,
        lat_col=lat_col,
        input_crs=input_crs,
        target_crs=DEFAULT_POINT_CRS,
    )
    origin_gdf, resolved_origin_id_col = _ensure_id_column(
        origin_gdf, origin_id_col, "origin"
    )
    total_origins = int(len(origin_gdf))
    group_columns = tuple(graph_group_cols or ())
    use_shared_graphs = bool(group_columns) or bool(share_graph_for_all_origins)
    batch_started = perf_counter()
    _log(
        "Starting batch drivesheds "
        f"(origins={total_origins}, max_travel_minutes={max_travel_minutes}, "
        f"h3_res={h3_res if h3_res is not None else 'profile_default'}, "
        f"use_shared_graphs={use_shared_graphs})"
    )

    origin_rows: List[gpd.GeoDataFrame] = []
    search_rows: List[Dict[str, Any]] = []
    polygon_rows: List[gpd.GeoDataFrame] = []
    cell_rows: List[gpd.GeoDataFrame] = []
    edge_rows: List[gpd.GeoDataFrame] = []
    graph_contexts: Dict[str, H3BatchGraphContext] = {}

    def _append_error_row(
        row: pd.Series,
        *,
        origin_id: Any,
        row_origin: Point,
        error_status: str,
        error_message: str,
        graph_group_id: Optional[str] = None,
    ) -> None:
        row_dict = dict(row)
        row_dict["status"] = error_status
        row_dict["error_message"] = error_message
        row_dict["geometry"] = row_origin
        if graph_group_id is not None:
            row_dict["graph_group_id"] = graph_group_id
        origin_rows.append(
            gpd.GeoDataFrame(
                [row_dict],
                geometry="geometry",
                crs=DEFAULT_POINT_CRS,
            )
        )

    def _append_success_outputs(
        row: pd.Series,
        *,
        origin_id: Any,
        result: dshed.DriveshedResult,
        graph_group_id: Optional[str] = None,
    ) -> None:
        base_attrs: Dict[str, Any] = {}
        if graph_group_id is not None:
            base_attrs["graph_group_id"] = graph_group_id

        origin_rows.append(
            gpd.GeoDataFrame(
                [
                    {
                        **dict(row),
                        **base_attrs,
                        "status": "ok",
                        "error_message": None,
                        "origin_h3_cell": result.origin_h3_cell,
                        "origin_h3_cell_graph": result.origin_h3_cell_graph,
                        "h3_res": result.h3_res,
                        "max_travel_minutes": result.max_travel_minutes,
                        "weight_attr": result.weight_attr,
                        "calibration_profile_name": result.calibration_profile_name,
                        "geometry": result.origin_point_wgs84,
                    }
                ],
                geometry="geometry",
                crs=DEFAULT_POINT_CRS,
            )
        )

        if result.search_polygon_wgs84 is not None and graph_group_id is None:
            search_rows.append(
                {
                    resolved_origin_id_col: origin_id,
                    "status": "ok",
                    "h3_res": result.h3_res,
                    "max_travel_minutes": result.max_travel_minutes,
                    "calibration_profile_name": result.calibration_profile_name,
                    "geometry": result.search_polygon_wgs84,
                }
            )

        shared_attrs = {
            **base_attrs,
            resolved_origin_id_col: origin_id,
            "status": "ok",
            "h3_res": result.h3_res,
            "max_travel_minutes": result.max_travel_minutes,
            "weight_attr": result.weight_attr,
            "calibration_profile_name": result.calibration_profile_name,
            "origin_h3_cell": result.origin_h3_cell,
            "origin_h3_cell_graph": result.origin_h3_cell_graph,
        }
        polygon_rows.append(
            _stamp_batch_metadata(
                result.driveshed_gdf,
                base_row=dict(row),
                extra_attrs=shared_attrs,
            )
        )
        cell_rows.append(
            _stamp_batch_metadata(
                result.reachable_cells_gdf,
                base_row=dict(row),
                extra_attrs=shared_attrs,
            )
        )
        edge_rows.append(
            _stamp_batch_metadata(
                result.reachable_edges_gdf,
                base_row=dict(row),
                extra_attrs=shared_attrs,
            )
        )

    def _run_one_origin(
        *,
        idx: int,
        row: pd.Series,
        graph_group_id: Optional[str] = None,
        context: Optional[H3BatchGraphContext] = None,
    ) -> None:
        origin_started = perf_counter()
        origin_id = row[resolved_origin_id_col]
        row_origin = row.geometry
        city_text = row.get("city")
        state_text = row.get("state")
        if pd.notna(city_text) and pd.notna(state_text):
            place_text = f"{city_text}, {state_text}"
        elif pd.notna(city_text):
            place_text = str(city_text)
        else:
            place_text = "unknown_place"
        pct_complete = (idx - 1) / total_origins * 100.0 if total_origins else 0.0
        _log(
            f"Building driveshed {idx}/{total_origins} ({pct_complete:.1f}% complete) "
            f"for origin_id={origin_id} ({place_text})"
        )
        try:
            if context is None:
                result = dshed.build_h3_driveshed_from_point(
                    row_origin,
                    max_travel_minutes=max_travel_minutes,
                    h3_res=h3_res,
                    weight_attr=weight_attr,
                    calibration_profile_name=calibration_profile_name,
                    calibration_profile_overrides=calibration_profile_overrides,
                    osm_cache_dir=osm_cache_dir,
                    osm_force_refresh=osm_force_refresh,
                    search_buffer_speed_mph=search_buffer_speed_mph,
                    search_buffer_factor=search_buffer_factor,
                    search_min_buffer_miles=search_min_buffer_miles,
                    snap_max_k=snap_max_k,
                    progress_label=f"origin_id={origin_id} ({place_text})",
                )
            else:
                result = dshed.build_h3_driveshed_from_point(
                    row_origin,
                    max_travel_minutes=max_travel_minutes,
                    h3_res=context.h3_res,
                    weight_attr=context.weight_attr,
                    calibration_profile_name=context.calibration_profile.name,
                    h3_graph=context.h3_graph,
                    G_osm=context.osm_graph,
                    search_polygon_wgs84=context.search_polygon_wgs84,
                    snap_max_k=snap_max_k,
                    progress_label=(
                        f"{graph_group_id} origin_id={origin_id} ({place_text})"
                        if graph_group_id
                        else f"origin_id={origin_id} ({place_text})"
                    ),
                )
        except Exception as exc:
            if not continue_on_error:
                raise
            elapsed_origin = perf_counter() - origin_started
            _log(
                f"Driveshed failed for origin_id={origin_id} "
                f"with {type(exc).__name__}: {exc} "
                f"(elapsed_sec={elapsed_origin:.1f})"
            )
            _append_error_row(
                row,
                origin_id=origin_id,
                row_origin=row_origin,
                error_status=f"error:{type(exc).__name__}",
                error_message=str(exc),
                graph_group_id=graph_group_id,
            )
            return

        elapsed_origin = perf_counter() - origin_started
        _log(
            f"Finished origin_id={origin_id} "
            f"(cells={len(result.reachable_cells_gdf)}, edges={len(result.reachable_edges_gdf)}, "
            f"elapsed_sec={elapsed_origin:.1f})"
        )
        _append_success_outputs(
            row,
            origin_id=origin_id,
            result=result,
            graph_group_id=graph_group_id,
        )

    if use_shared_graphs:
        grouped_frames = list(_iter_groups(origin_gdf, group_columns if group_columns else None))
        for group_idx, (graph_group_id, group_key, group_frame) in enumerate(
            grouped_frames, start=1
        ):
            group_started = perf_counter()
            _log(
                f"Preparing shared driveshed graph {group_idx}/{len(grouped_frames)} "
                f"(graph_group_id={graph_group_id}, origins={len(group_frame)})"
            )
            try:
                search_polygon_wgs84 = build_points_convex_hull_search_polygon(
                    group_frame,
                    geometry_col="geometry",
                    input_crs=DEFAULT_POINT_CRS,
                    buffer_miles=shared_graph_buffer_miles,
                )
                context = build_calibrated_h3_graph_for_points(
                    group_frame,
                    graph_group_id=graph_group_id,
                    group_key=group_key,
                    group_columns=group_columns,
                    geometry_col="geometry",
                    input_crs=DEFAULT_POINT_CRS,
                    search_polygon_wgs84=search_polygon_wgs84,
                    h3_res=h3_res,
                    calibration_profile_name=calibration_profile_name,
                    calibration_profile_overrides=calibration_profile_overrides,
                    weight_attr=weight_attr,
                    osm_cache_dir=osm_cache_dir,
                    osm_force_refresh=osm_force_refresh,
                )
            except Exception as exc:
                if not continue_on_error:
                    raise
                _log(
                    f"Shared graph preparation failed for graph_group_id={graph_group_id} "
                    f"with {type(exc).__name__}: {exc} "
                    f"(elapsed_sec={perf_counter() - group_started:.1f})"
                )
                for _, row in group_frame.iterrows():
                    _append_error_row(
                        row,
                        origin_id=row[resolved_origin_id_col],
                        row_origin=row.geometry,
                        error_status=f"error:{type(exc).__name__}",
                        error_message=str(exc),
                        graph_group_id=graph_group_id,
                    )
                continue

            graph_contexts[graph_group_id] = context
            search_row = _build_search_polygon_row(
                graph_group_id=graph_group_id,
                group_columns=tuple(group_columns),
                group_key=group_key,
                context=context,
            )
            search_row["status"] = "ok"
            search_row["max_travel_minutes"] = float(max_travel_minutes)
            search_row["shared_graph_buffer_miles"] = float(shared_graph_buffer_miles)
            search_rows.append(search_row)
            _log(
                f"Shared driveshed graph ready for graph_group_id={graph_group_id} "
                f"(elapsed_sec={perf_counter() - group_started:.1f})"
            )
            for _, row in group_frame.iterrows():
                _run_one_origin(
                    idx=int(row.name) + 1,
                    row=row,
                    graph_group_id=graph_group_id,
                    context=context,
                )
    else:
        for idx, (_, row) in enumerate(origin_gdf.iterrows(), start=1):
            _run_one_origin(idx=idx, row=row)

    origins_out = _concat_gdfs(origin_rows)
    search_out = _rows_to_gdf(search_rows, geometry_col="geometry")
    polygon_out = _concat_gdfs(polygon_rows)
    cells_out = _concat_gdfs(cell_rows)
    edges_out = _concat_gdfs(edge_rows)

    upsampled_cells_out = gpd.GeoDataFrame(
        columns=["geometry"], geometry="geometry", crs=DEFAULT_POINT_CRS
    )
    upsampled_polygons_out = gpd.GeoDataFrame(
        columns=["geometry"], geometry="geometry", crs=DEFAULT_POINT_CRS
    )
    if upsampled_target_resolutions and not cells_out.empty:
        upsample_started = perf_counter()
        _log(
            "Upsampling driveshed cells "
            f"to target resolutions {tuple(upsampled_target_resolutions)}"
        )
        upsampled_cells_out = dshed.aggregate_driveshed_cells_to_parent_layers(
            cells_out,
            target_resolutions=upsampled_target_resolutions,
            h3_cell_col="h3_cell",
        )
        if not upsampled_cells_out.empty:
            upsampled_polygons_out = dshed.dissolve_upsampled_driveshed_cells(
                upsampled_cells_out,
                source_cells_gdf=cells_out,
                h3_cell_col="h3_cell",
            )
        _log(
            f"Upsampling finished (cells={len(upsampled_cells_out)}, "
            f"polygons={len(upsampled_polygons_out)}, "
            f"elapsed_sec={perf_counter() - upsample_started:.1f})"
        )

    elapsed_batch = perf_counter() - batch_started
    _log(
        "Batch drivesheds complete "
        f"(origin_rows={len(origins_out)}, polygons={len(polygon_out)}, "
        f"cells={len(cells_out)}, edges={len(edges_out)}, "
        f"elapsed_sec={elapsed_batch:.1f})"
    )

    return BatchDriveshedResult(
        origins_gdf=origins_out,
        search_polygons_gdf=search_out,
        driveshed_polygons_gdf=polygon_out,
        driveshed_cells_gdf=cells_out,
        driveshed_edges_gdf=edges_out,
        driveshed_cells_upsampled_gdf=upsampled_cells_out,
        driveshed_polygons_upsampled_gdf=upsampled_polygons_out,
        graph_contexts=graph_contexts,
    )


def _coerce_point_value(value: Any) -> Point:
    if isinstance(value, Point):
        return value
    if isinstance(value, str):
        parsed = shapely_wkt.loads(value)
        if isinstance(parsed, Point):
            return parsed
    raise ValueError("Expected POINT geometry or WKT POINT string.")


def _ensure_point_geometry_series(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    bad_rows: List[int] = []
    for idx, geom in enumerate(out.geometry.tolist()):
        if geom is None or not isinstance(geom, Point) or geom.is_empty:
            bad_rows.append(idx)
    if bad_rows:
        raise ValueError(
            f"All geometries must be non-empty Point geometries. Bad row positions: {bad_rows[:10]}"
        )
    return out


def _resolve_point_series(
    frame: pd.DataFrame,
    *,
    geometry_col: Optional[str],
    lon_col: str,
    lat_col: str,
    input_crs: str,
    target_crs: str,
    source_gdf: Optional[gpd.GeoDataFrame],
    allow_active_geometry: bool,
) -> pd.Series:
    if geometry_col and geometry_col in frame.columns:
        temp = gpd.GeoDataFrame(
            frame.copy(),
            geometry=frame[geometry_col].apply(_coerce_point_value),
            crs=input_crs,
        )
        if str(temp.crs).lower() != str(target_crs).lower():
            temp = temp.to_crs(target_crs)
        return temp.geometry.reset_index(drop=True)

    if allow_active_geometry and source_gdf is not None:
        temp = source_gdf.copy()
        if temp.crs is None:
            temp = temp.set_crs(input_crs)
        if str(temp.crs).lower() != str(target_crs).lower():
            temp = temp.to_crs(target_crs)
        return temp.geometry.reset_index(drop=True)

    missing = [col for col in (lon_col, lat_col) if col not in frame.columns]
    if missing:
        raise ValueError(
            f"Could not resolve point columns. Missing: {missing}. "
            f"Expected geometry_col='{geometry_col}' or lon/lat columns."
        )

    temp = gpd.GeoDataFrame(
        frame.copy(),
        geometry=gpd.points_from_xy(frame[lon_col], frame[lat_col]),
        crs=input_crs,
    )
    if str(temp.crs).lower() != str(target_crs).lower():
        temp = temp.to_crs(target_crs)
    return temp.geometry.reset_index(drop=True)


def _ensure_id_column(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    requested_id_col: Optional[str],
    prefix: str,
) -> Tuple[pd.DataFrame | gpd.GeoDataFrame, str]:
    out = frame.copy()
    if requested_id_col and requested_id_col in out.columns:
        return out, requested_id_col
    generated_col = f"{prefix}_row_id"
    if generated_col in out.columns:
        return out, generated_col
    out[generated_col] = range(1, len(out) + 1)
    return out, generated_col


def _iter_groups(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    group_cols: Optional[Sequence[str]],
) -> Iterable[Tuple[str, Tuple[Any, ...], pd.DataFrame | gpd.GeoDataFrame]]:
    if not group_cols:
        yield "all", (), frame
        return

    grouped = frame.groupby(list(group_cols), dropna=False, sort=False)
    for idx, (group_key, group_frame) in enumerate(grouped, start=1):
        key_tuple = group_key if isinstance(group_key, tuple) else (group_key,)
        yield f"group_{idx:05d}", tuple(key_tuple), group_frame.copy()


def _subset_by_group_key(
    frame: pd.DataFrame | gpd.GeoDataFrame,
    group_cols: Optional[Sequence[str]],
    group_key: Tuple[Any, ...],
) -> pd.DataFrame | gpd.GeoDataFrame:
    if not group_cols:
        return frame.copy()
    mask = pd.Series(True, index=frame.index)
    for col, value in zip(group_cols, group_key):
        if pd.isna(value):
            mask = mask & frame[col].isna()
        else:
            mask = mask & (frame[col] == value)
    return frame.loc[mask].copy()


def _build_search_polygon_row(
    *,
    graph_group_id: str,
    group_columns: Tuple[str, ...],
    group_key: Tuple[Any, ...],
    context: H3BatchGraphContext,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "graph_group_id": graph_group_id,
        "h3_res": int(context.h3_res),
        "weight_attr": context.weight_attr,
        "report_weight_attr": context.report_weight_attr,
        "calibration_profile_name": context.calibration_profile.name,
        "geometry": context.search_polygon_wgs84,
    }
    for col, value in zip(group_columns, group_key):
        row[col] = value
    return row


def _solve_h3_route(
    *,
    h3_graph: nx.Graph,
    origin_wgs84: Point,
    destination_wgs84: Point,
    h3_res: int,
    weight_attr: str,
    report_weight_attr: str,
    snap_max_k: int,
) -> Dict[str, Any]:
    origin_cell = h3.latlng_to_cell(
        float(origin_wgs84.y), float(origin_wgs84.x), h3_res
    )
    destination_cell = h3.latlng_to_cell(
        float(destination_wgs84.y), float(destination_wgs84.x), h3_res
    )
    graph_nodes: set[str] = set(h3_graph.nodes)

    origin_cell_graph = hnetx.snap_cell_to_graph(
        origin_cell, graph_nodes, max_k=snap_max_k
    )
    destination_cell_graph = hnetx.snap_cell_to_graph(
        destination_cell, graph_nodes, max_k=snap_max_k
    )
    if origin_cell_graph is None or destination_cell_graph is None:
        return {
            "status": "snap_failed",
            "error_message": None,
            "path_cells": None,
            "travel_time_sec": None,
            "travel_time_postcalibrated_sec": None,
            "travel_miles": None,
            "origin_cell": origin_cell,
            "destination_cell": destination_cell,
            "origin_cell_graph": origin_cell_graph,
            "destination_cell_graph": destination_cell_graph,
            "geometry": None,
        }

    try:
        path_cells = nx.shortest_path(
            h3_graph,
            source=origin_cell_graph,
            target=destination_cell_graph,
            weight=weight_attr,
        )
        travel_time_sec = float(
            nx.shortest_path_length(
                h3_graph,
                source=origin_cell_graph,
                target=destination_cell_graph,
                weight=weight_attr,
            )
        )
    except nx.NetworkXNoPath:
        return {
            "status": "no_path",
            "error_message": None,
            "path_cells": None,
            "travel_time_sec": None,
            "travel_time_postcalibrated_sec": None,
            "travel_miles": None,
            "origin_cell": origin_cell,
            "destination_cell": destination_cell,
            "origin_cell_graph": origin_cell_graph,
            "destination_cell_graph": destination_cell_graph,
            "geometry": None,
        }

    travel_time_postcalibrated_sec = hnetx.path_weight_sum(
        h3_graph,
        path_cells,
        weight_attr=report_weight_attr,
    )
    return {
        "status": "ok",
        "error_message": None,
        "path_cells": list(path_cells),
        "travel_time_sec": travel_time_sec,
        "travel_time_postcalibrated_sec": float(travel_time_postcalibrated_sec),
        "travel_miles": _path_distance_miles(h3_graph, path_cells),
        "origin_cell": origin_cell,
        "destination_cell": destination_cell,
        "origin_cell_graph": origin_cell_graph,
        "destination_cell_graph": destination_cell_graph,
        "geometry": hnetx.h3_path_to_linestring(list(path_cells)),
    }


def _path_distance_miles(h3_graph: nx.Graph, path_cells: Sequence[str]) -> float:
    total = 0.0
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        edge_data = h3_graph[a][b]
        if edge_data.get("centroid_dist_miles") is not None:
            total += float(edge_data["centroid_dist_miles"])
        elif edge_data.get("centroid_dist_m") is not None:
            total += float(edge_data["centroid_dist_m"]) / hnetx.METERS_PER_MILE
    return float(total)


def _build_route_hex_rows(
    *,
    route_id: Any,
    path_cells: Sequence[str],
    graph_group_id: str,
    extra_attrs: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    extra = dict(extra_attrs or {})
    for step, cell in enumerate(path_cells):
        rows.append(
            {
                "route_id": route_id,
                "graph_group_id": graph_group_id,
                "step": int(step),
                "h3_cell": str(cell),
                "geometry": _build_h3_cell_polygon(str(cell)),
                **extra,
            }
        )
    return rows


def _build_route_hex_rows_from_gdf(
    gdf: gpd.GeoDataFrame,
    *,
    route_id_col: str,
    path_col: str,
    graph_group_id: str,
    extra_cols: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in gdf.iterrows():
        path_value = row.get(path_col)
        if not isinstance(path_value, str) or not path_value.strip():
            continue
        extra = {col: row.get(col) for col in extra_cols if col in row.index}
        rows.extend(
            _build_route_hex_rows(
                route_id=row[route_id_col],
                path_cells=path_value.split("|"),
                graph_group_id=graph_group_id,
                extra_attrs={route_id_col: row[route_id_col], **extra},
            )
        )
    return rows


def _build_h3_cell_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def _stamp_batch_metadata(
    gdf: gpd.GeoDataFrame,
    *,
    base_row: Mapping[str, Any],
    extra_attrs: Mapping[str, Any],
) -> gpd.GeoDataFrame:
    out = gdf.copy()
    for col, value in base_row.items():
        if col == "geometry":
            continue
        out[col] = value
    for col, value in extra_attrs.items():
        out[col] = value
    return out


def _rows_to_gdf(
    rows: Sequence[Mapping[str, Any]],
    *,
    geometry_col: str,
) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame(
            columns=[geometry_col], geometry=geometry_col, crs=DEFAULT_POINT_CRS
        )
    return gpd.GeoDataFrame(list(rows), geometry=geometry_col, crs=DEFAULT_POINT_CRS)


def _concat_gdfs(frames: Sequence[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    non_empty = [frame for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        return gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=DEFAULT_POINT_CRS
        )
    return gpd.GeoDataFrame(
        pd.concat(non_empty, ignore_index=True),
        geometry="geometry",
        crs=DEFAULT_POINT_CRS,
    )
