from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import inf
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pyproj
from platformdirs import user_cache_dir
from shapely.geometry import LineString, Point, Polygon, box
from shapely import wkt as shapely_wkt

METERS_PER_MILE: float = 1609.344


def miles_to_meters(miles: float) -> float:
    return float(miles) * METERS_PER_MILE


def _default_cache_dir() -> Path:
    return Path(user_cache_dir("osmh3nx"))


DEFAULT_CACHE_DIR: str = str(_default_cache_dir())


def _graph_cache_key(
    *,
    polygon_wgs84: Polygon,
    simplify: bool,
    network_type: str,
) -> str:
    key_raw = "|".join(
        [
            f"network_type={network_type}",
            f"simplify={int(bool(simplify))}",
            polygon_wgs84.wkt,
        ]
    )
    return hashlib.sha1(key_raw.encode("utf-8")).hexdigest()


def _to_float_or_keep(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return value
        try:
            return float(txt)
        except Exception:
            return value
    return value


def _normalize_loaded_graph(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    # GraphML loads frequently return numeric attrs as strings.
    for _, data in G.nodes(data=True):
        if "x" in data:
            data["x"] = _to_float_or_keep(data.get("x"))
        if "y" in data:
            data["y"] = _to_float_or_keep(data.get("y"))

    for _, _, _, data in G.edges(keys=True, data=True):
        for key in ("length", "speed_kph", "travel_time"):
            if key in data:
                data[key] = _to_float_or_keep(data.get(key))
        geom = data.get("geometry")
        if isinstance(geom, str):
            try:
                parsed = shapely_wkt.loads(geom)
                if isinstance(parsed, LineString) and not parsed.is_empty:
                    data["geometry"] = parsed
            except Exception:
                pass
    return G


def _coords_look_like_wgs84(G: nx.MultiDiGraph, sample_n: int = 1000) -> bool:
    checked = 0
    for _, data in G.nodes(data=True):
        x_raw = data.get("x")
        y_raw = data.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            x = float(x_raw)
            y = float(y_raw)
        except Exception:
            return False
        if x < -180.0 or x > 180.0 or y < -90.0 or y > 90.0:
            return False
        checked += 1
        if checked >= sample_n:
            break
    return checked > 0


def _graph_overlaps_polygon_bbox(G: nx.MultiDiGraph, polygon_wgs84: Polygon) -> bool:
    xs: List[float] = []
    ys: List[float] = []
    for _, data in G.nodes(data=True):
        x_raw = data.get("x")
        y_raw = data.get("y")
        if x_raw is None or y_raw is None:
            continue
        try:
            xs.append(float(x_raw))
            ys.append(float(y_raw))
        except Exception:
            continue
    if not xs or not ys:
        return False

    g_minx, g_maxx = min(xs), max(xs)
    g_miny, g_maxy = min(ys), max(ys)
    p_minx, p_miny, p_maxx, p_maxy = polygon_wgs84.bounds

    overlaps_x = max(g_minx, p_minx) <= min(g_maxx, p_maxx)
    overlaps_y = max(g_miny, p_miny) <= min(g_maxy, p_maxy)
    return bool(overlaps_x and overlaps_y)


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
    cache_dir: Optional[str] = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> nx.MultiDiGraph:
    """
    Download and enrich a drivable OSM network for a polygon.
    Uses GraphML cache when available.
    """
    network_type = "drive"
    graph_key = _graph_cache_key(
        polygon_wgs84=polygon_wgs84,
        simplify=simplify,
        network_type=network_type,
    )
    cache_path: Optional[Path] = None
    base_cache: Optional[Path] = None
    if cache_dir is not None:
        base_cache = Path(cache_dir) if cache_dir else _default_cache_dir()
        if not base_cache.is_absolute():
            base_cache = Path.cwd() / base_cache
        base_cache.mkdir(parents=True, exist_ok=True)
        cache_path = base_cache / f"osm_drive_{graph_key}.graphml"

    loaded_from_cache = False
    if cache_path is not None and cache_path.exists() and not force_refresh:
        try:
            G = ox.load_graphml(
                filepath=str(cache_path),
                node_dtypes={"x": float, "y": float},
                edge_dtypes={
                    "length": float,
                    "speed_kph": float,
                    "travel_time": float,
                    "geometry": shapely_wkt.loads,
                },
            )
        except TypeError:
            G = ox.load_graphml(filepath=str(cache_path))
        G = _normalize_loaded_graph(G)
        loaded_from_cache = True
    else:
        prior_cache_folder = ox.settings.cache_folder
        try:
            if base_cache is not None:
                ox.settings.cache_folder = str(base_cache)
            G = ox.graph_from_polygon(polygon_wgs84, network_type=network_type, simplify=simplify)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            if cache_path is not None:
                ox.save_graphml(G, filepath=str(cache_path))
        finally:
            ox.settings.cache_folder = prior_cache_folder

    # Guardrail: if cache declares WGS84 but coordinates are not WGS84-like,
    # rebuild to avoid downstream CRS/projection corruption.
    crs_raw = str(G.graph.get("crs") or "")
    cache_invalid = False
    if loaded_from_cache and "4326" in crs_raw and not _coords_look_like_wgs84(G):
        cache_invalid = True
    if loaded_from_cache and not _graph_overlaps_polygon_bbox(G, polygon_wgs84):
        cache_invalid = True

    if cache_invalid:
        prior_cache_folder = ox.settings.cache_folder
        try:
            if base_cache is not None:
                ox.settings.cache_folder = str(base_cache)
            G = ox.graph_from_polygon(polygon_wgs84, network_type=network_type, simplify=simplify)
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            if cache_path is not None:
                ox.save_graphml(G, filepath=str(cache_path))
        finally:
            ox.settings.cache_folder = prior_cache_folder

    # Ensure required attrs exist on loaded graphs from any older cache versions.
    if not all("speed_kph" in data for _, _, _, data in G.edges(keys=True, data=True)):
        G = ox.add_edge_speeds(G)
    if not all("travel_time" in data for _, _, _, data in G.edges(keys=True, data=True)):
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
    Uses projected nearest-node lookup to avoid unprojected NN optional deps.
    Returns route geometry in EPSG:4326.
    """
    G_proj = ox.project_graph(G)
    crs_proj = G_proj.graph.get("crs")
    if crs_proj is None:
        raise ValueError("Projected OSM graph is missing CRS.")
    to_proj = pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
    ox_x, ox_y = to_proj.transform(float(origin_wgs84.x), float(origin_wgs84.y))
    dx_x, dx_y = to_proj.transform(float(destination_wgs84.x), float(destination_wgs84.y))

    origin_node = ox.distance.nearest_nodes(
        G_proj,
        X=float(ox_x),
        Y=float(ox_y),
    )
    destination_node = ox.distance.nearest_nodes(
        G_proj,
        X=float(dx_x),
        Y=float(dx_y),
    )

    route_nodes = nx.shortest_path(
        G_proj,
        source=origin_node,
        target=destination_node,
        weight=weight_attr,
    )
    geom, travel_time_sec, length_m, route_edges = _build_route_geometry_and_metrics(
        G_proj,
        route_nodes,
        weight_attr=weight_attr,
    )
    geom_wgs84 = gpd.GeoSeries([geom], crs=crs_proj).to_crs("EPSG:4326").iloc[0]
    return OSMRouteResult(
        origin_node=origin_node,
        destination_node=destination_node,
        route_nodes=list(route_nodes),
        route_edges=route_edges,
        travel_time_sec=float(travel_time_sec),
        length_m=float(length_m),
        geometry=geom_wgs84,
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
