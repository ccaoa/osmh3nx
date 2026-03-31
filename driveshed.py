from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point, Polygon

import h3

import network_h3
import network_osm


SECONDS_PER_MINUTE: float = 60.0
DEFAULT_H3_WEIGHT_ATTR: str = "observed_step_time_raw_sec"


@dataclass
class DriveshedResult:
    origin_point_wgs84: Point
    origin_h3_cell: str
    origin_h3_cell_graph: str
    h3_res: int
    max_travel_minutes: float
    cutoff_seconds: float
    weight_attr: str
    search_polygon_wgs84: Optional[Polygon]
    osm_graph: Optional[nx.MultiDiGraph]
    h3_graph: nx.Graph
    distances_seconds: Dict[str, float]
    paths: Dict[str, List[str]]
    reachable_cells_gdf: gpd.GeoDataFrame
    reachable_edges_gdf: gpd.GeoDataFrame
    driveshed_gdf: gpd.GeoDataFrame


def _ensure_wgs84_point(origin: Any) -> Point:
    if isinstance(origin, Point):
        return origin

    if isinstance(origin, gpd.GeoSeries):
        if len(origin) != 1:
            raise ValueError("GeoSeries origin must contain exactly one geometry.")
        geom = origin.iloc[0]
        crs = origin.crs
    elif isinstance(origin, gpd.GeoDataFrame):
        if len(origin) != 1:
            raise ValueError("GeoDataFrame origin must contain exactly one row.")
        geom = origin.geometry.iloc[0]
        crs = origin.crs
    else:
        raise TypeError("origin must be a shapely Point, GeoSeries, or GeoDataFrame.")

    if geom is None or geom.is_empty:
        raise ValueError("Origin geometry is empty.")
    if not isinstance(geom, Point):
        raise ValueError("Origin geometry must be a Point.")
    if crs is None:
        raise ValueError("Origin GeoSeries/GeoDataFrame must have a CRS.")
    if str(crs).lower() in {"epsg:4326", "wgs84"}:
        return geom

    origin_gdf = gpd.GeoDataFrame({"row_id": [0]}, geometry=[geom], crs=crs).to_crs("EPSG:4326")
    return origin_gdf.geometry.iloc[0]


def estimate_driveshed_buffer_miles(
    *,
    max_travel_minutes: float = 15.0,
    buffer_speed_mph: float = 60.0,
    buffer_factor: float = 1.4,
    min_buffer_miles: float = 2.0,
) -> float:
    """
    Estimate a download radius for the OSM graph around a single origin point.

    The multiplier intentionally overshoots the straight-line equivalent because
    road networks are circuitous and a driveshed can branch in many directions.
    """
    if max_travel_minutes <= 0:
        raise ValueError("max_travel_minutes must be > 0.")
    if buffer_speed_mph <= 0:
        raise ValueError("buffer_speed_mph must be > 0.")
    if buffer_factor < 1.0:
        raise ValueError("buffer_factor must be >= 1.0.")
    if min_buffer_miles <= 0:
        raise ValueError("min_buffer_miles must be > 0.")

    travel_hours = float(max_travel_minutes) / SECONDS_PER_MINUTE / 60.0
    estimate = travel_hours * float(buffer_speed_mph) * float(buffer_factor)
    return float(max(float(min_buffer_miles), estimate))


def build_driveshed_search_polygon(
    origin: Any,
    *,
    max_travel_minutes: float = 15.0,
    buffer_speed_mph: float = 60.0,
    buffer_factor: float = 1.4,
    min_buffer_miles: float = 2.0,
) -> Polygon:
    """
    Build a local search polygon around a single origin point for OSM download.
    """
    origin_wgs84 = _ensure_wgs84_point(origin)
    buffer_miles = estimate_driveshed_buffer_miles(
        max_travel_minutes=max_travel_minutes,
        buffer_speed_mph=buffer_speed_mph,
        buffer_factor=buffer_factor,
        min_buffer_miles=min_buffer_miles,
    )

    origin_gdf = gpd.GeoDataFrame({"row_id": [0]}, geometry=[origin_wgs84], crs="EPSG:4326")
    proj_crs = origin_gdf.estimate_utm_crs() or "EPSG:3857"
    origin_proj = origin_gdf.to_crs(proj_crs)
    buffer_m = network_osm.miles_to_meters(buffer_miles)
    polygon_proj = origin_proj.geometry.iloc[0].buffer(buffer_m)
    polygon_wgs84 = gpd.GeoSeries([polygon_proj], crs=proj_crs).to_crs("EPSG:4326").iloc[0]
    return polygon_wgs84


def _validate_weight_attr(h3_graph: nx.Graph, weight_attr: str) -> None:
    if h3_graph.number_of_edges() == 0:
        raise ValueError("h3_graph has no edges.")
    _, _, edge_data = next(iter(h3_graph.edges(data=True)))
    if weight_attr not in edge_data:
        raise ValueError(f"weight_attr '{weight_attr}' not present on H3 graph edges.")


def _build_cell_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def _build_reachable_cells_gdf(
    *,
    distances: Dict[str, float],
    paths: Dict[str, List[str]],
    origin_cell_graph: str,
) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for cell, time_sec in sorted(distances.items(), key=lambda item: (float(item[1]), item[0])):
        path = paths.get(cell, [])
        rows.append(
            {
                "h3_cell": cell,
                "travel_time_sec": float(time_sec),
                "travel_time_minutes": float(time_sec) / SECONDS_PER_MINUTE,
                "path_n_cells": int(len(path)),
                "is_origin_cell_graph": bool(cell == origin_cell_graph),
                "geometry": _build_cell_polygon(cell),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _edge_is_within_cutoff(
    *,
    h3_graph: nx.Graph,
    edge: Tuple[str, str],
    distances: Dict[str, float],
    cutoff_seconds: float,
    weight_attr: str,
) -> bool:
    a, b = edge
    edge_weight = float(h3_graph[a][b].get(weight_attr, 0.0))
    dist_a = distances.get(a)
    dist_b = distances.get(b)
    if dist_a is None or dist_b is None:
        return False
    if h3_graph.is_directed():
        return bool(dist_a + edge_weight <= cutoff_seconds + 1e-9)
    return bool(
        (dist_a + edge_weight <= cutoff_seconds + 1e-9)
        or (dist_b + edge_weight <= cutoff_seconds + 1e-9)
    )


def _build_reachable_edges_gdf(
    *,
    h3_graph: nx.Graph,
    distances: Dict[str, float],
    cutoff_seconds: float,
    weight_attr: str,
) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for a, b, data in h3_graph.edges(data=True):
        if not _edge_is_within_cutoff(
            h3_graph=h3_graph,
            edge=(a, b),
            distances=distances,
            cutoff_seconds=cutoff_seconds,
            weight_attr=weight_attr,
        ):
            continue

        lat_a, lng_a = h3.cell_to_latlng(a)
        lat_b, lng_b = h3.cell_to_latlng(b)
        rows.append(
            {
                "from_h3_cell": a,
                "to_h3_cell": b,
                "is_directed_graph": bool(h3_graph.is_directed()),
                "travel_time_sec": float(data.get(weight_attr, 0.0)),
                "travel_time_minutes": float(data.get(weight_attr, 0.0)) / SECONDS_PER_MINUTE,
                "observed_step_time_raw_sec": data.get("observed_step_time_raw_sec"),
                "step_time_floored_sec": data.get("step_time_floored_sec"),
                "step_time_route_sec": data.get("step_time_route_sec"),
                "step_time_postcalibrated_sec": data.get("step_time_postcalibrated_sec"),
                "centroid_dist_miles": data.get("centroid_dist_miles"),
                "floor_applied": data.get("floor_applied"),
                "geometry": LineString([(lng_a, lat_a), (lng_b, lat_b)]),
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _dissolve_reachable_cells(reachable_cells_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if reachable_cells_gdf.empty:
        return gpd.GeoDataFrame(
            {
                "n_cells": [0],
                "geometry": [None],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )

    try:
        driveshed_geom = reachable_cells_gdf.geometry.union_all()
    except AttributeError:
        driveshed_geom = reachable_cells_gdf.unary_union

    max_time_sec = float(reachable_cells_gdf["travel_time_sec"].max())
    return gpd.GeoDataFrame(
        {
            "n_cells": [int(len(reachable_cells_gdf))],
            "max_travel_time_sec": [max_time_sec],
            "max_travel_time_minutes": [max_time_sec / SECONDS_PER_MINUTE],
            "geometry": [driveshed_geom],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )


def build_h3_driveshed_from_point(
    origin: Any,
    *,
    max_travel_minutes: float = 15.0,
    h3_res: int = 10,
    weight_attr: str = DEFAULT_H3_WEIGHT_ATTR,
    h3_graph: Optional[nx.Graph] = None,
    G_osm: Optional[nx.MultiDiGraph] = None,
    search_polygon_wgs84: Optional[Polygon] = None,
    osm_cache_dir: Optional[str] = "cache",
    osm_force_refresh: bool = False,
    search_buffer_speed_mph: float = 60.0,
    search_buffer_factor: float = 1.4,
    search_min_buffer_miles: float = 2.0,
    sample_miles: Optional[float] = 0.1,
    sample_meters: Optional[float] = None,
    combine_parallel: str = "min",
    enforce_min_step_time: bool = True,
    v_max_mph: Optional[float] = 50.0,
    v_max_kph: Optional[float] = None,
    floor_speed_source: str = "vmax",
    min_osm_speed_mph: Optional[float] = 10.0,
    min_osm_speed_kph: Optional[float] = None,
    directional: bool = True,
    preserve_way_geometry: bool = True,
    way_cell_refine_max_depth: int = 18,
    route_weight_attr: str = "travel_time_route",
    route_floor_penalty_weight: float = 0.35,
    report_weight_attr: str = "travel_time_postcalibrated",
    report_floor_penalty_weight: float = 1.0,
    snap_max_k: int = 10,
) -> DriveshedResult:
    """
    Build an H3 driveshed from a single origin point.

    Dijkstra is still the correct algorithm here. A driveshed is a single-source
    shortest-path problem with a travel-time cutoff, and all edge weights are
    positive travel times.
    """
    if max_travel_minutes <= 0:
        raise ValueError("max_travel_minutes must be > 0.")
    if h3_res < 0:
        raise ValueError("h3_res must be >= 0.")
    if snap_max_k < 0:
        raise ValueError("snap_max_k must be >= 0.")

    origin_wgs84 = _ensure_wgs84_point(origin)
    cutoff_seconds = float(max_travel_minutes) * SECONDS_PER_MINUTE

    if h3_graph is None:
        if G_osm is None:
            if search_polygon_wgs84 is None:
                search_polygon_wgs84 = build_driveshed_search_polygon(
                    origin_wgs84,
                    max_travel_minutes=max_travel_minutes,
                    buffer_speed_mph=search_buffer_speed_mph,
                    buffer_factor=search_buffer_factor,
                    min_buffer_miles=search_min_buffer_miles,
                )
            G_osm = network_osm.download_osm_drive_graph_for_polygon(
                search_polygon_wgs84,
                cache_dir=osm_cache_dir,
                force_refresh=osm_force_refresh,
            )

        h3_graph = network_h3.build_h3_travel_graph_from_osm(
            G_osm,
            h3_res,
            weight_attr="travel_time",
            sample_miles=sample_miles,
            sample_meters=sample_meters,
            combine_parallel=combine_parallel,
            enforce_min_step_time=enforce_min_step_time,
            v_max_mph=v_max_mph,
            v_max_kph=v_max_kph,
            floor_speed_source=floor_speed_source,
            min_osm_speed_mph=min_osm_speed_mph,
            min_osm_speed_kph=min_osm_speed_kph,
            directional=directional,
            preserve_way_geometry=preserve_way_geometry,
            way_cell_refine_max_depth=way_cell_refine_max_depth,
            route_weight_attr=route_weight_attr,
            route_floor_penalty_weight=route_floor_penalty_weight,
            report_weight_attr=report_weight_attr,
            report_floor_penalty_weight=report_floor_penalty_weight,
        )
    else:
        graph_h3_res = h3_graph.graph.get("h3_res")
        if graph_h3_res is not None and int(graph_h3_res) != int(h3_res):
            raise ValueError(
                f"Provided h3_graph has h3_res={graph_h3_res}, but h3_res={h3_res} was requested."
            )

    _validate_weight_attr(h3_graph, weight_attr)

    graph_nodes = set(h3_graph.nodes)
    origin_cell = h3.latlng_to_cell(origin_wgs84.y, origin_wgs84.x, h3_res)
    origin_cell_graph = network_h3.snap_cell_to_graph(origin_cell, graph_nodes, max_k=snap_max_k)
    if origin_cell_graph is None:
        raise ValueError("Origin cell could not be snapped to the H3 graph within snap_max_k.")

    distances, paths = nx.single_source_dijkstra(
        h3_graph,
        source=origin_cell_graph,
        cutoff=cutoff_seconds,
        weight=weight_attr,
    )
    distances = {str(cell): float(time_sec) for cell, time_sec in distances.items()}
    paths = {str(cell): [str(step) for step in path] for cell, path in paths.items()}

    reachable_cells_gdf = _build_reachable_cells_gdf(
        distances=distances,
        paths=paths,
        origin_cell_graph=origin_cell_graph,
    )
    reachable_edges_gdf = _build_reachable_edges_gdf(
        h3_graph=h3_graph,
        distances=distances,
        cutoff_seconds=cutoff_seconds,
        weight_attr=weight_attr,
    )
    driveshed_gdf = _dissolve_reachable_cells(reachable_cells_gdf)

    return DriveshedResult(
        origin_point_wgs84=origin_wgs84,
        origin_h3_cell=origin_cell,
        origin_h3_cell_graph=origin_cell_graph,
        h3_res=int(h3_res),
        max_travel_minutes=float(max_travel_minutes),
        cutoff_seconds=float(cutoff_seconds),
        weight_attr=str(weight_attr),
        search_polygon_wgs84=search_polygon_wgs84,
        osm_graph=G_osm,
        h3_graph=h3_graph,
        distances_seconds=distances,
        paths=paths,
        reachable_cells_gdf=reachable_cells_gdf,
        reachable_edges_gdf=reachable_edges_gdf,
        driveshed_gdf=driveshed_gdf,
    )


def write_driveshed_result_to_gpkg(
    result: DriveshedResult,
    output_path: str | Path,
    *,
    layer_prefix: str = "driveshed",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    origin_gdf = gpd.GeoDataFrame(
        {
            "origin_h3_cell": [result.origin_h3_cell],
            "origin_h3_cell_graph": [result.origin_h3_cell_graph],
            "h3_res": [result.h3_res],
            "max_travel_minutes": [result.max_travel_minutes],
            "weight_attr": [result.weight_attr],
        },
        geometry=[result.origin_point_wgs84],
        crs="EPSG:4326",
    )
    origin_gdf.to_file(output_path, layer=f"{layer_prefix}_origin", driver="GPKG")
    result.reachable_cells_gdf.to_file(output_path, layer=f"{layer_prefix}_cells", driver="GPKG")
    result.reachable_edges_gdf.to_file(output_path, layer=f"{layer_prefix}_edges", driver="GPKG")
    result.driveshed_gdf.to_file(output_path, layer=f"{layer_prefix}_polygon", driver="GPKG")

    if result.search_polygon_wgs84 is not None:
        search_gdf = gpd.GeoDataFrame(
            {"h3_res": [result.h3_res]},
            geometry=[result.search_polygon_wgs84],
            crs="EPSG:4326",
        )
        search_gdf.to_file(output_path, layer=f"{layer_prefix}_search_polygon", driver="GPKG")

    return output_path


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build an H3 driveshed from a single origin point.")
    parser.add_argument("--lon", type=float, required=True, help="Origin longitude in WGS84.")
    parser.add_argument("--lat", type=float, required=True, help="Origin latitude in WGS84.")
    parser.add_argument("--h3-res", type=int, default=10, help="H3 resolution. Default: 10.")
    parser.add_argument(
        "--max-travel-minutes",
        type=float,
        default=15.0,
        help="Maximum travel time in minutes. Default: 15.",
    )
    parser.add_argument(
        "--weight-attr",
        type=str,
        default=DEFAULT_H3_WEIGHT_ATTR,
        help="H3 edge attribute used for driveshed travel time. Default: observed_step_time_raw_sec.",
    )
    parser.add_argument(
        "--output-gpkg",
        type=str,
        default=None,
        help="Optional output GeoPackage path.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    origin = Point(float(args.lon), float(args.lat))
    result = build_h3_driveshed_from_point(
        origin,
        max_travel_minutes=float(args.max_travel_minutes),
        h3_res=int(args.h3_res),
        weight_attr=str(args.weight_attr),
    )

    print("Origin H3 cell:", result.origin_h3_cell)
    print("Snapped origin H3 cell:", result.origin_h3_cell_graph)
    print("Reachable H3 cells:", len(result.reachable_cells_gdf))
    print("Reachable H3 edges:", len(result.reachable_edges_gdf))

    if args.output_gpkg:
        output_path = write_driveshed_result_to_gpkg(result, args.output_gpkg)
        print("Wrote driveshed GeoPackage:", output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
