from __future__ import annotations

from dataclasses import dataclass
from math import inf
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, Point, Polygon, box

METERS_PER_MILE: float = 1609.344


def miles_to_meters(miles: float) -> float:
    return float(miles) * METERS_PER_MILE


@dataclass(frozen=True)
class OSMRouteResult:
    origin_node: Any
    destination_node: Any
    route_nodes: List[Any]
    route_edges: List[Tuple[Any, Any, Any]]
    travel_time_sec: float
    length_m: float
    geometry: LineString


def build_pair_square_and_buffer(
    origin_wgs84: Point,
    destination_wgs84: Point,
    *,
    buffer_miles: float = 10.0,
    min_square_side_m: float = 1000.0,
) -> Tuple[Polygon, Polygon]:
    """
    Build a square around the OD bounding box, then buffer it by buffer_miles.
    Returns (square_wgs84, buffered_square_wgs84).
    """
    if buffer_miles <= 0:
        raise ValueError("buffer_miles must be > 0.")
    if min_square_side_m <= 0:
        raise ValueError("min_square_side_m must be > 0.")

    pts = gpd.GeoDataFrame(
        {"row_id": [0, 1]},
        geometry=[origin_wgs84, destination_wgs84],
        crs="EPSG:4326",
    )
    proj_crs = pts.estimate_utm_crs()
    if proj_crs is None:
        proj_crs = "EPSG:3857"

    pts_proj = pts.to_crs(proj_crs)
    minx, miny, maxx, maxy = [float(x) for x in pts_proj.total_bounds]
    side = max(maxx - minx, maxy - miny, float(min_square_side_m))
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0
    half = side / 2.0

    square_proj = box(cx - half, cy - half, cx + half, cy + half)
    buffer_proj = square_proj.buffer(miles_to_meters(buffer_miles))

    square_wgs84 = gpd.GeoSeries([square_proj], crs=proj_crs).to_crs("EPSG:4326").iloc[0]
    buffered_wgs84 = gpd.GeoSeries([buffer_proj], crs=proj_crs).to_crs("EPSG:4326").iloc[0]
    return square_wgs84, buffered_wgs84


def download_osm_drive_graph_for_polygon(
    polygon_wgs84: Polygon,
    *,
    simplify: bool = True,
) -> nx.MultiDiGraph:
    """
    Download and enrich a drivable OSM network for a polygon.
    """
    G = ox.graph_from_polygon(polygon_wgs84, network_type="drive", simplify=simplify)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G


def _select_best_parallel_edge_data(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    *,
    weight_attr: str,
) -> Tuple[Any, Dict[str, Any]]:
    edge_lookup = G.get_edge_data(u, v)
    if not edge_lookup:
        raise ValueError(f"No edge data found between nodes {u} and {v}.")

    best_key: Optional[Any] = None
    best_data: Optional[Dict[str, Any]] = None
    best_weight = inf

    for key, data in edge_lookup.items():
        weight_raw = data.get(weight_attr)
        try:
            weight = float(weight_raw)
        except Exception:
            weight = inf

        if weight < best_weight:
            best_weight = weight
            best_key = key
            best_data = data

    if best_data is None:
        best_key = sorted(edge_lookup.keys(), key=lambda x: str(x))[0]
        best_data = edge_lookup[best_key]

    return best_key, dict(best_data)


def _edge_geometry_from_data(G: nx.MultiDiGraph, u: Any, v: Any, data: Dict[str, Any]) -> LineString:
    geom = data.get("geometry")
    if isinstance(geom, LineString) and not geom.is_empty:
        return geom

    ux = G.nodes[u].get("x")
    uy = G.nodes[u].get("y")
    vx = G.nodes[v].get("x")
    vy = G.nodes[v].get("y")
    if ux is None or uy is None or vx is None or vy is None:
        raise ValueError(f"Missing node coordinates for fallback edge geometry between {u} and {v}.")
    return LineString([(float(ux), float(uy)), (float(vx), float(vy))])


def _orient_segment_to_previous(
    segment: Sequence[Tuple[float, float]],
    prev_xy: Tuple[float, float],
) -> List[Tuple[float, float]]:
    if len(segment) <= 1:
        return list(segment)

    start = segment[0]
    end = segment[-1]
    d_start = (prev_xy[0] - start[0]) ** 2 + (prev_xy[1] - start[1]) ** 2
    d_end = (prev_xy[0] - end[0]) ** 2 + (prev_xy[1] - end[1]) ** 2
    if d_end < d_start:
        return list(reversed(segment))
    return list(segment)


def _build_route_geometry_and_metrics(
    G: nx.MultiDiGraph,
    route_nodes: Sequence[Any],
    *,
    weight_attr: str = "travel_time",
) -> Tuple[LineString, float, float, List[Tuple[Any, Any, Any]]]:
    if len(route_nodes) < 2:
        raise ValueError("Route must contain at least two nodes.")

    route_coords: List[Tuple[float, float]] = []
    route_edges: List[Tuple[Any, Any, Any]] = []
    travel_time_sec = 0.0
    length_m = 0.0

    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        key, data = _select_best_parallel_edge_data(G, u, v, weight_attr=weight_attr)
        route_edges.append((u, v, key))
        try:
            travel_time_sec += float(data.get(weight_attr, 0.0))
        except Exception:
            pass
        try:
            length_m += float(data.get("length", 0.0))
        except Exception:
            pass

        seg_line = _edge_geometry_from_data(G, u, v, data)
        seg_coords = list(seg_line.coords)
        if not route_coords:
            route_coords.extend(seg_coords)
            continue

        seg_coords = _orient_segment_to_previous(seg_coords, route_coords[-1])
        if route_coords[-1] == seg_coords[0]:
            route_coords.extend(seg_coords[1:])
        else:
            route_coords.extend(seg_coords)

    if len(route_coords) < 2:
        raise ValueError("Route geometry could not be constructed.")

    return LineString(route_coords), float(travel_time_sec), float(length_m), route_edges


def route_between_points_on_graph(
    G: nx.MultiDiGraph,
    *,
    origin_wgs84: Point,
    destination_wgs84: Point,
    weight_attr: str = "travel_time",
) -> OSMRouteResult:
    """
    Route one OD pair on an OSM graph.
    """
    origin_node = ox.distance.nearest_nodes(
        G,
        X=float(origin_wgs84.x),
        Y=float(origin_wgs84.y),
    )
    destination_node = ox.distance.nearest_nodes(
        G,
        X=float(destination_wgs84.x),
        Y=float(destination_wgs84.y),
    )

    route_nodes = nx.shortest_path(G, source=origin_node, target=destination_node, weight=weight_attr)
    geom, travel_time_sec, length_m, route_edges = _build_route_geometry_and_metrics(
        G,
        route_nodes,
        weight_attr=weight_attr,
    )
    return OSMRouteResult(
        origin_node=origin_node,
        destination_node=destination_node,
        route_nodes=list(route_nodes),
        route_edges=route_edges,
        travel_time_sec=float(travel_time_sec),
        length_m=float(length_m),
        geometry=geom,
    )


def graph_to_context_gdfs(
    G: nx.MultiDiGraph,
    *,
    pair_id: int,
    city: str,
    state: str,
    category: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Convert one graph to node/edge GeoDataFrames and stamp pair metadata.
    """
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(
        G,
        nodes=True,
        edges=True,
        fill_edge_geometry=True,
    )
    nodes = nodes_gdf.reset_index()
    edges = edges_gdf.reset_index()

    for frame in (nodes, edges):
        frame["pair_id"] = int(pair_id)
        frame["city"] = str(city)
        frame["state"] = str(state)
        frame["category"] = str(category)

    if "length" in edges.columns:
        edges["length_miles"] = edges["length"].astype(float) / METERS_PER_MILE

    nodes = gpd.GeoDataFrame(nodes, geometry="geometry", crs=nodes_gdf.crs).to_crs("EPSG:4326")
    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=edges_gdf.crs).to_crs("EPSG:4326")
    return nodes, edges
