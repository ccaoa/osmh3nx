from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import h3
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

try:
    from . import network_h3 as hnetx
    from . import network_osm as onetx
except ImportError:
    import network_h3 as hnetx
    import network_osm as onetx


@dataclass(frozen=True)
class CalibrationConfig:
    h3_resolutions: Tuple[int, ...] = (7, 8, 9, 10)
    osm_weight_attr: str = "travel_time"
    h3_weight_attr: str = "travel_time_route"
    bbox_buffer_miles: float = 15.0
    sample_miles: float = 0.1
    combine_parallel: str = "min"
    directional: bool = True
    enforce_min_step_time: bool = True
    v_max_mph: float = 50.0
    floor_speed_source: str = "vmax"
    min_osm_speed_mph: float = 10.0 / hnetx.KM_PER_MILE
    route_weight_attr: str = "travel_time_route"
    route_floor_penalty_weight: float = 0.35
    report_weight_attr: str = "travel_time_postcalibrated"
    report_floor_penalty_weight: float = 1.0
    shape_corridor_meters: float = 100.0
    snap_k: int = 10
    osm_cache_dir: Optional[str] = "cache"
    osm_force_refresh: bool = False  #  True  #


def _required_columns() -> List[str]:
    return ["count", "city", "state", "category", "origin", "destination"]


def load_od_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in _required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration CSV: {missing}")

    out = df.copy()
    out["pair_id"] = out["count"].astype(int)
    out["origin_geom"] = out["origin"].apply(wkt.loads)
    out["destination_geom"] = out["destination"].apply(wkt.loads)

    bad_origin = out["origin_geom"].apply(lambda g: not isinstance(g, Point))
    bad_dest = out["destination_geom"].apply(lambda g: not isinstance(g, Point))
    if bool(bad_origin.any()) or bool(bad_dest.any()):
        raise ValueError("Origin and destination WKT must parse to POINT geometries.")

    out = out.sort_values("pair_id").reset_index(drop=True)
    return out


def build_od_points_gdf(od_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in od_pairs.iterrows():
        base = {
            "pair_id": int(r["pair_id"]),
            "count": int(r["count"]),
            "city": r["city"],
            "state": r["state"],
            "category": r["category"],
            "origin": r["origin"],
            "destination": r["destination"],
        }
        rows.append(
            {
                **base,
                "point_role": "origin",
                "point_wkt": r["origin"],
                "geometry": r["origin_geom"],
            }
        )
        rows.append(
            {
                **base,
                "point_role": "destination",
                "point_wkt": r["destination"],
                "geometry": r["destination_geom"],
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _h3_cell_to_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def _h3_path_distance_miles(h3_graph: nx.Graph, path_cells: Sequence[str]) -> float:
    total_miles = 0.0
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        edge_data = h3_graph[a][b]
        dist_miles_raw = edge_data.get("centroid_dist_miles")
        if dist_miles_raw is not None:
            total_miles += float(dist_miles_raw)
            continue
        dist_m_raw = edge_data.get("centroid_dist_m")
        if dist_m_raw is not None:
            total_miles += float(dist_m_raw) / hnetx.METERS_PER_MILE
    return float(total_miles)


def _safe_percent_delta(estimate: Optional[float], truth: Optional[float]) -> Optional[float]:
    if estimate is None or truth is None:
        return None
    if not pd.notna(estimate) or not pd.notna(truth):
        return None
    truth_val = float(truth)
    if truth_val == 0.0:
        return None
    return (float(estimate) - truth_val) / truth_val * 100.0


def _safe_abs(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return abs(float(value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _harmonic_mean_ratio(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    a_val = float(a)
    b_val = float(b)
    if a_val <= 0 or b_val <= 0:
        return 0.0
    return float((2.0 * a_val * b_val) / (a_val + b_val))


def _default_shape_metrics(*, status: str, corridor_m: float) -> Dict[str, Any]:
    return {
        "shape_status": status,
        "shape_osm_corridor_m": float(corridor_m),
        "osm_covered_by_h3_ratio": None,
        "osm_covered_by_h3_pct": None,
        "h3_within_osm_corridor_ratio": None,
        "h3_within_osm_corridor_pct": None,
        "shape_f1_ratio": None,
        "shape_f1_pct": None,
        "shape_hausdorff_m": None,
    }


def _compute_shape_metrics(
    *,
    osm_route_wgs84: Optional[LineString],
    h3_route_wgs84: Optional[LineString],
    h3_path_cells: Sequence[str],
    origin_wgs84: Point,
    destination_wgs84: Point,
    corridor_m: float,
) -> Dict[str, Any]:
    if corridor_m <= 0:
        return _default_shape_metrics(status="error:bad_corridor", corridor_m=corridor_m)
    if osm_route_wgs84 is None or h3_route_wgs84 is None:
        return _default_shape_metrics(status="missing_geometry", corridor_m=corridor_m)
    if len(h3_path_cells) < 2:
        return _default_shape_metrics(status="short_h3_path", corridor_m=corridor_m)

    try:
        h3_union_wgs84 = unary_union([_h3_cell_to_polygon(c) for c in h3_path_cells])

        seed = gpd.GeoDataFrame(
            {"kind": ["origin", "destination"]},
            geometry=[origin_wgs84, destination_wgs84],
            crs="EPSG:4326",
        )
        metric_crs = seed.estimate_utm_crs() or "EPSG:3857"

        geoms_wgs84 = gpd.GeoDataFrame(
            {"kind": ["osm_line", "h3_line", "h3_cells"]},
            geometry=[osm_route_wgs84, h3_route_wgs84, h3_union_wgs84],
            crs="EPSG:4326",
        )
        geoms_metric = geoms_wgs84.to_crs(metric_crs)

        osm_line_m = geoms_metric.loc[geoms_metric["kind"] == "osm_line", "geometry"].iloc[0]
        h3_line_m = geoms_metric.loc[geoms_metric["kind"] == "h3_line", "geometry"].iloc[0]
        h3_cells_m = geoms_metric.loc[geoms_metric["kind"] == "h3_cells", "geometry"].iloc[0]

        osm_len_m = float(osm_line_m.length)
        h3_len_m = float(h3_line_m.length)
        if osm_len_m <= 0 or h3_len_m <= 0:
            return _default_shape_metrics(status="zero_length", corridor_m=corridor_m)

        osm_covered_len_m = float(osm_line_m.intersection(h3_cells_m).length)
        osm_covered_ratio = _clamp01(osm_covered_len_m / osm_len_m)

        osm_corridor = osm_line_m.buffer(float(corridor_m))
        h3_in_corridor_len_m = float(h3_line_m.intersection(osm_corridor).length)
        h3_in_corridor_ratio = _clamp01(h3_in_corridor_len_m / h3_len_m)

        f1_ratio = _harmonic_mean_ratio(osm_covered_ratio, h3_in_corridor_ratio)
        hausdorff_m = float(h3_line_m.hausdorff_distance(osm_line_m))

        return {
            "shape_status": "ok",
            "shape_osm_corridor_m": float(corridor_m),
            "osm_covered_by_h3_ratio": osm_covered_ratio,
            "osm_covered_by_h3_pct": float(osm_covered_ratio * 100.0),
            "h3_within_osm_corridor_ratio": h3_in_corridor_ratio,
            "h3_within_osm_corridor_pct": float(h3_in_corridor_ratio * 100.0),
            "shape_f1_ratio": f1_ratio,
            "shape_f1_pct": float(f1_ratio * 100.0) if f1_ratio is not None else None,
            "shape_hausdorff_m": hausdorff_m,
        }
    except Exception as exc:
        return _default_shape_metrics(status=f"error:{type(exc).__name__}", corridor_m=corridor_m)


def solve_h3_route_for_od(
    *,
    h3_graph: nx.Graph,
    origin_wgs84: Point,
    destination_wgs84: Point,
    h3_res: int,
    weight_attr: str,
    report_weight_attr: str,
    snap_k: int,
) -> Dict[str, Any]:
    origin_cell = h3.latlng_to_cell(float(origin_wgs84.y), float(origin_wgs84.x), h3_res)
    destination_cell = h3.latlng_to_cell(float(destination_wgs84.y), float(destination_wgs84.x), h3_res)

    graph_nodes: set[str] = set(h3_graph.nodes)
    origin_snap = hnetx.snap_cell_to_graph(origin_cell, graph_nodes, max_k=snap_k)
    destination_snap = hnetx.snap_cell_to_graph(destination_cell, graph_nodes, max_k=snap_k)
    if origin_snap is None or destination_snap is None:
        return {
            "status": "snap_failed",
            "path_cells": None,
            "travel_time_sec": None,
            "travel_time_postcalibrated_sec": None,
            "travel_miles": None,
            "origin_cell": origin_cell,
            "destination_cell": destination_cell,
            "origin_cell_snap": origin_snap,
            "destination_cell_snap": destination_snap,
            "geometry": None,
        }

    path_cells = nx.shortest_path(
        h3_graph,
        source=origin_snap,
        target=destination_snap,
        weight=weight_attr,
    )
    travel_time_sec = float(
        nx.shortest_path_length(
            h3_graph,
            source=origin_snap,
            target=destination_snap,
            weight=weight_attr,
        )
    )
    travel_time_postcalibrated_sec = hnetx.path_weight_sum(
        h3_graph,
        path_cells,
        weight_attr=report_weight_attr,
    )
    travel_miles = _h3_path_distance_miles(h3_graph, path_cells)
    geometry = hnetx.h3_path_to_linestring(list(path_cells))
    return {
        "status": "ok",
        "path_cells": list(path_cells),
        "travel_time_sec": float(travel_time_sec),
        "travel_time_postcalibrated_sec": float(travel_time_postcalibrated_sec),
        "travel_miles": float(travel_miles),
        "origin_cell": origin_cell,
        "destination_cell": destination_cell,
        "origin_cell_snap": origin_snap,
        "destination_cell_snap": destination_snap,
        "geometry": geometry,
    }


def _to_gdf(rows: List[Dict[str, Any]], *, geometry_col: str = "geometry") -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame(columns=[geometry_col], geometry=geometry_col, crs="EPSG:4326")
    return gpd.GeoDataFrame(rows, geometry=geometry_col, crs="EPSG:4326")


def write_layers_to_gpkg(
    gpkg_path: str,
    *,
    layers: Sequence[Tuple[str, gpd.GeoDataFrame]],
) -> List[str]:
    parent = os.path.dirname(gpkg_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    written_layers: List[str] = []
    mode = "w"
    for layer_name, gdf in layers:
        if gdf.empty:
            continue
        gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode)
        written_layers.append(layer_name)
        mode = "a"
    return written_layers


def run_h3_osm_calibration(
    *,
    csv_path: str,
    output_gpkg_path: str,
    config: CalibrationConfig,
) -> Dict[str, Any]:
    od_pairs = load_od_pairs(csv_path)
    od_points = build_od_points_gdf(od_pairs)

    metrics_rows: List[Dict[str, Any]] = []
    osm_route_rows: List[Dict[str, Any]] = []
    h3_route_rows: List[Dict[str, Any]] = []
    h3_hex_rows: List[Dict[str, Any]] = []
    square_rows: List[Dict[str, Any]] = []
    buffered_rows: List[Dict[str, Any]] = []

    context_nodes_frames: List[gpd.GeoDataFrame] = []
    context_edges_frames: List[gpd.GeoDataFrame] = []

    for i, row in od_pairs.iterrows():
        pair_id = int(row["pair_id"])
        city = str(row["city"])
        state = str(row["state"])
        category = str(row["category"])
        origin = row["origin_geom"]
        destination = row["destination_geom"]
        print(f"[{i + 1}/{len(od_pairs)}] pair_id={pair_id} city={city}, {state}")

        square_poly, buffer_poly = onetx.build_pair_square_and_buffer(
            origin,
            destination,
            buffer_miles=config.bbox_buffer_miles,
        )
        square_rows.append(
            {
                "pair_id": pair_id,
                "city": city,
                "state": state,
                "category": category,
                "geometry": square_poly,
            }
        )
        buffered_rows.append(
            {
                "pair_id": pair_id,
                "city": city,
                "state": state,
                "category": category,
                "geometry": buffer_poly,
            }
        )

        osm_status = "ok"
        osm_time_sec: Optional[float] = None
        osm_miles: Optional[float] = None
        osm_geometry: Optional[LineString] = None
        osm_graph_nodes: Optional[int] = None
        osm_graph_edges: Optional[int] = None
        G_osm: Optional[nx.MultiDiGraph] = None

        try:
            G_osm = onetx.download_osm_drive_graph_for_polygon(
                buffer_poly,
                cache_dir=config.osm_cache_dir,
                force_refresh=config.osm_force_refresh,
            )
            osm_graph_nodes = G_osm.number_of_nodes()
            osm_graph_edges = G_osm.number_of_edges()

            nodes_ctx, edges_ctx = onetx.graph_to_context_gdfs(
                G_osm,
                pair_id=pair_id,
                city=city,
                state=state,
                category=category,
            )
            context_nodes_frames.append(nodes_ctx)
            context_edges_frames.append(edges_ctx)

            osm_result = onetx.route_between_points_on_graph(
                G_osm,
                origin_wgs84=origin,
                destination_wgs84=destination,
                weight_attr=config.osm_weight_attr,
            )
            osm_time_sec = float(osm_result.travel_time_sec)
            osm_miles = float(osm_result.length_m / hnetx.METERS_PER_MILE)
            osm_geometry = osm_result.geometry
            osm_route_rows.append(
                {
                    "pair_id": pair_id,
                    "count": pair_id,
                    "city": city,
                    "state": state,
                    "category": category,
                    "origin": row["origin"],
                    "destination": row["destination"],
                    "osm_origin_node": osm_result.origin_node,
                    "osm_destination_node": osm_result.destination_node,
                    "osm_n_nodes": len(osm_result.route_nodes),
                    "osm_n_edges": len(osm_result.route_edges),
                    "osm_travel_time_sec": osm_time_sec,
                    "osm_travel_miles": osm_miles,
                    "geometry": osm_result.geometry,
                }
            )
        except nx.NetworkXNoPath:
            osm_status = "no_path"
        except Exception as exc:
            osm_status = f"error:{type(exc).__name__}"

        for h3_res in config.h3_resolutions:
            h3_status = "ok"
            h3_time_sec: Optional[float] = None
            h3_time_postcalibrated_sec: Optional[float] = None
            h3_miles: Optional[float] = None
            shape_metrics = _default_shape_metrics(status="not_computed", corridor_m=config.shape_corridor_meters)
            h3_graph_nodes: Optional[int] = None
            h3_graph_edges: Optional[int] = None
            h3_cells_count: Optional[int] = None
            origin_cell: Optional[str] = None
            destination_cell: Optional[str] = None
            origin_cell_snap: Optional[str] = None
            destination_cell_snap: Optional[str] = None
            h3_graph_directional: Optional[bool] = None

            if G_osm is None:
                h3_status = "osm_graph_missing"
                shape_metrics = _default_shape_metrics(
                    status="osm_graph_missing",
                    corridor_m=config.shape_corridor_meters,
                )
            else:
                try:
                    H_h3 = hnetx.build_h3_travel_graph_from_osm(
                        G_osm,
                        h3_res=h3_res,
                        weight_attr=config.osm_weight_attr,
                        sample_miles=config.sample_miles,
                        combine_parallel=config.combine_parallel,
                        directional=config.directional,
                        enforce_min_step_time=config.enforce_min_step_time,
                        v_max_mph=config.v_max_mph,
                        floor_speed_source=config.floor_speed_source,
                        min_osm_speed_mph=config.min_osm_speed_mph,
                        route_weight_attr=config.route_weight_attr,
                        route_floor_penalty_weight=config.route_floor_penalty_weight,
                        report_weight_attr=config.report_weight_attr,
                        report_floor_penalty_weight=config.report_floor_penalty_weight,
                    )
                    h3_graph_nodes = H_h3.number_of_nodes()
                    h3_graph_edges = H_h3.number_of_edges()
                    h3_graph_directional = bool(H_h3.graph.get("directional", H_h3.is_directed()))

                    h3_result = solve_h3_route_for_od(
                        h3_graph=H_h3,
                        origin_wgs84=origin,
                        destination_wgs84=destination,
                        h3_res=h3_res,
                        weight_attr=config.h3_weight_attr,
                        report_weight_attr=config.report_weight_attr,
                        snap_k=config.snap_k,
                    )
                    h3_status = str(h3_result["status"])
                    origin_cell = h3_result["origin_cell"]
                    destination_cell = h3_result["destination_cell"]
                    origin_cell_snap = h3_result["origin_cell_snap"]
                    destination_cell_snap = h3_result["destination_cell_snap"]

                    if h3_result["status"] == "ok":
                        h3_time_sec = float(h3_result["travel_time_sec"])
                        h3_time_postcalibrated_sec = float(h3_result["travel_time_postcalibrated_sec"])
                        h3_miles = float(h3_result["travel_miles"])
                        path_cells = list(h3_result["path_cells"])
                        h3_cells_count = len(path_cells)
                        if osm_status == "ok" and osm_geometry is not None:
                            shape_metrics = _compute_shape_metrics(
                                osm_route_wgs84=osm_geometry,
                                h3_route_wgs84=h3_result["geometry"],
                                h3_path_cells=path_cells,
                                origin_wgs84=origin,
                                destination_wgs84=destination,
                                corridor_m=config.shape_corridor_meters,
                            )
                        else:
                            shape_metrics = _default_shape_metrics(
                                status="osm_not_ok",
                                corridor_m=config.shape_corridor_meters,
                            )
                        h3_route_rows.append(
                            {
                                "pair_id": pair_id,
                                "count": pair_id,
                                "city": city,
                                "state": state,
                                "category": category,
                                "origin": row["origin"],
                                "destination": row["destination"],
                                "h3_res": int(h3_res),
                                "h3_directional": h3_graph_directional,
                                "h3_n_cells": h3_cells_count,
                                "h3_origin_cell": origin_cell,
                                "h3_destination_cell": destination_cell,
                                "h3_origin_cell_snap": origin_cell_snap,
                                "h3_destination_cell_snap": destination_cell_snap,
                                "h3_path": "|".join(path_cells),
                                "h3_travel_time_sec": h3_time_sec,
                                "h3_travel_time_postcalibrated_sec": h3_time_postcalibrated_sec,
                                "h3_travel_miles": h3_miles,
                                "geometry": h3_result["geometry"],
                            }
                        )
                        for step, cell in enumerate(path_cells):
                            h3_hex_rows.append(
                                {
                                    "pair_id": pair_id,
                                    "count": pair_id,
                                    "city": city,
                                    "state": state,
                                    "category": category,
                                    "origin": row["origin"],
                                    "destination": row["destination"],
                                    "h3_res": int(h3_res),
                                    "h3_directional": h3_graph_directional,
                                    "step": int(step),
                                    "h3_cell": cell,
                                    "geometry": _h3_cell_to_polygon(cell),
                                }
                            )
                    else:
                        shape_metrics = _default_shape_metrics(
                            status=f"h3_{h3_result['status']}",
                            corridor_m=config.shape_corridor_meters,
                        )
                except nx.NetworkXNoPath:
                    h3_status = "no_path"
                    shape_metrics = _default_shape_metrics(
                        status="h3_no_path",
                        corridor_m=config.shape_corridor_meters,
                    )
                except Exception as exc:
                    h3_status = f"error:{type(exc).__name__}"
                    shape_metrics = _default_shape_metrics(
                        status=f"h3_error:{type(exc).__name__}",
                        corridor_m=config.shape_corridor_meters,
                    )

            metrics_rows.append(
                {
                    "pair_id": pair_id,
                    "count": pair_id,
                    "city": city,
                    "state": state,
                    "category": category,
                    "origin": row["origin"],
                    "destination": row["destination"],
                    "h3_res": int(h3_res),
                    "status_osm": osm_status,
                    "status_h3": h3_status,
                    "osm_travel_time_sec": osm_time_sec,
                    "h3_travel_time_sec": h3_time_sec,
                    "h3_travel_time_postcalibrated_sec": h3_time_postcalibrated_sec,
                    "time_error_sec": (h3_time_sec - osm_time_sec) if h3_time_sec is not None and osm_time_sec is not None else None,
                    "time_error_pct": _safe_percent_delta(h3_time_sec, osm_time_sec),
                    "abs_time_error": _safe_abs(_safe_percent_delta(h3_time_sec, osm_time_sec)),
                    "time_postcalibrated_error_sec": (h3_time_postcalibrated_sec - osm_time_sec) if h3_time_postcalibrated_sec is not None and osm_time_sec is not None else None,
                    "time_postcalibrated_error_pct": _safe_percent_delta(h3_time_postcalibrated_sec, osm_time_sec),
                    "abs_time_postcalibrated_error": _safe_abs(_safe_percent_delta(h3_time_postcalibrated_sec, osm_time_sec)),
                    "osm_travel_miles": osm_miles,
                    "h3_travel_miles": h3_miles,
                    "distance_error_miles": (h3_miles - osm_miles) if h3_miles is not None and osm_miles is not None else None,
                    "distance_error_pct": _safe_percent_delta(h3_miles, osm_miles),
                    "abs_dist_error": _safe_abs(_safe_percent_delta(h3_miles, osm_miles)),
                    "osm_graph_nodes": osm_graph_nodes,
                    "osm_graph_edges": osm_graph_edges,
                    "h3_graph_nodes": h3_graph_nodes,
                    "h3_graph_edges": h3_graph_edges,
                    "h3_directional": h3_graph_directional,
                    "h3_n_cells": h3_cells_count,
                    "h3_origin_cell": origin_cell,
                    "h3_destination_cell": destination_cell,
                    "h3_origin_cell_snap": origin_cell_snap,
                    "h3_destination_cell_snap": destination_cell_snap,
                    **shape_metrics,
                }
            )

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["pair_id", "h3_res"]).reset_index(drop=True)
    osm_routes_gdf = _to_gdf(osm_route_rows)
    h3_routes_gdf = _to_gdf(h3_route_rows)
    h3_hexes_gdf = _to_gdf(h3_hex_rows)
    squares_gdf = _to_gdf(square_rows)
    buffers_gdf = _to_gdf(buffered_rows)

    if context_nodes_frames:
        context_nodes_gdf = gpd.GeoDataFrame(
            pd.concat(context_nodes_frames, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        context_nodes_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    if context_edges_frames:
        context_edges_gdf = gpd.GeoDataFrame(
            pd.concat(context_edges_frames, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
    else:
        context_edges_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    written_layers = write_layers_to_gpkg(
        output_gpkg_path,
        layers=[
            ("od_points", od_points),
            ("osm_routes_truth", osm_routes_gdf),
            ("h3_routes", h3_routes_gdf),
            ("h3_route_hexes", h3_hexes_gdf),
            ("pair_square_bbox", squares_gdf),
            ("pair_buffer_bbox", buffers_gdf),
            ("osm_context_nodes", context_nodes_gdf),
            ("osm_context_edges", context_edges_gdf),
        ],
    )

    metrics_csv_path = str(Path(output_gpkg_path).with_suffix("")) + "_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)

    return {
        "written_layers": written_layers,
        "output_gpkg_path": output_gpkg_path,
        "metrics_csv_path": metrics_csv_path,
        "metrics_df": metrics_df,
        "od_points": od_points,
        "osm_routes_gdf": osm_routes_gdf,
        "h3_routes_gdf": h3_routes_gdf,
        "h3_hexes_gdf": h3_hexes_gdf,
        "context_nodes_gdf": context_nodes_gdf,
        "context_edges_gdf": context_edges_gdf,
    }


if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.log_console = True

    vintage = 7
    output_dir = os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing")
    csv_file = os.path.join(os.path.dirname(__file__), "osm_scale_calibration.csv")
    output_gpkg = os.path.join(output_dir, f"h3_osm_calibration_vintage{vintage}.gpkg")

    cfg = CalibrationConfig(
        h3_resolutions=(7, 8, 9, 10),
        bbox_buffer_miles=5.0,
        sample_miles=0.1,
        combine_parallel="min",
        directional=True,
        enforce_min_step_time=True,
        v_max_mph=50.0,
        floor_speed_source="vmax",
        min_osm_speed_mph=15,
        route_weight_attr="travel_time_route",
        route_floor_penalty_weight=0.35,
        report_weight_attr="travel_time_postcalibrated",
        report_floor_penalty_weight=1.0,
        shape_corridor_meters=100.0,
        snap_k=10,
        osm_force_refresh=False,
    )

    result = run_h3_osm_calibration(
        csv_path=csv_file,
        output_gpkg_path=output_gpkg,
        config=cfg,
    )
    print("Calibration complete.")
    print("GPKG:", result["output_gpkg_path"])
    print("Metrics CSV:", result["metrics_csv_path"])
    print("Layers written:", result["written_layers"])
