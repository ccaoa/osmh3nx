"""
I have the most recent versions of the python packages h3, networkx, and osmnx downloaded. I want to use OSM roads to do network analysis by using h3 hexagons at different scales. This would help me not have to tax the computational load nearly as much as doing a purely route-based network analysis with OSM roads directly.

This is my inspiration. Please look at this link for an idea. I want to build something similar in python (the back-end processing, not necessarily the interactive map in a Jupyter type space, cool though that is).
https://observablehq.com/@nrabinowitz/h3-travel-times

# TODO Read these other sources
#  - https://github.com/nmandery/rout3serv?tab=readme-ov-file & https://github.com/nmandery/rout3serv/blob/main/crates/rout3serv/README.md
#  - Especially https://bytes.swiggy.com/the-osm-distance-service-part-1-evaluation-metrics-and-routing-configurations-6e8686ca814f
#  - https://heigit.org/another-milestone-for-ohsomenow-h3-hexagon-maps-at-high-and-low-resolution/


I want to look at a test case in Radford City and Montgomery County, Virginia to help begin to build this. Write me initial code that pulls OSM street data, h3 polygons at a county-appropriate scale, and a function that accepts two point geopandas datasets. The end goal is to see which point from one dataset is closest to the point in the other dataset by h3 network analysis. The unique ID of the compare dataset's closest point should be written to a new column in points from the source dataset.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Set
import re
import warnings

import networkx as nx
import numpy as np, pandas as pd
import osmnx as ox
from shapely.geometry import Point, Polygon, LineString, MultiPolygon
import pyproj
import geopandas as gpd
import h3
from h3 import LatLngMultiPoly, LatLngPoly

METERS_PER_MILE: float = 1609.344
KM_PER_MILE: float = 1.609344
SECONDS_PER_HOUR: float = 3600.0


def _meters_to_miles(meters: float) -> float:
    return float(meters) / METERS_PER_MILE


def _miles_to_meters(miles: float) -> float:
    return float(miles) * METERS_PER_MILE


def _kph_to_mph(kph: float) -> float:
    return float(kph) / KM_PER_MILE


def _mph_to_kph(mph: float) -> float:
    return float(mph) * KM_PER_MILE


def _resolve_distance_meters(
    *,
    miles: Optional[float],
    meters: Optional[float],
    default_meters: float,
) -> float:
    if miles is not None and meters is not None:
        raise ValueError("Specify only one of miles or meters, not both.")
    if miles is not None:
        out = _miles_to_meters(float(miles))
    elif meters is not None:
        out = float(meters)
    else:
        out = float(default_meters)
    if out <= 0:
        raise ValueError("Distance must be > 0.")
    return out


def _resolve_speed_kph(
    *,
    mph: Optional[float],
    kph: Optional[float],
    default_kph: float,
) -> float:
    if mph is not None and kph is not None:
        raise ValueError("Specify only one of mph or kph, not both.")
    if mph is not None:
        out = _mph_to_kph(float(mph))
    elif kph is not None:
        out = float(kph)
    else:
        out = float(default_kph)
    if out <= 0:
        raise ValueError("Speed must be > 0.")
    return out


def _coerce_speed_to_kph(speed_raw: Any) -> Optional[float]:
    # OSMnx edge attributes can be scalar, list, or strings like "25 mph".
    vals_kph: List[float] = []
    candidates = speed_raw if isinstance(speed_raw, (list, tuple, np.ndarray, pd.Series)) else [speed_raw]
    for item in candidates:
        if item is None:
            continue
        if isinstance(item, str):
            txt = item.strip().lower()
            match = re.search(r"[-+]?\d*\.?\d+", txt)
            if match is None:
                continue
            val = float(match.group(0))
            if "mph" in txt:
                val = _mph_to_kph(val)
        else:
            try:
                val = float(item)
            except Exception:
                continue
        if np.isfinite(val) and val > 0:
            vals_kph.append(val)
    if not vals_kph:
        return None
    return float(np.median(vals_kph))


def download_osm_graph_drive(polygon_wgs84: Polygon) -> nx.MultiDiGraph:
    """
    Download a drivable street network from OSM within the given polygon.
    Adds edge speeds and travel times.
    """
    # network_type can be "drive", "walk", etc.
    G = ox.graph_from_polygon(polygon_wgs84, network_type="drive", simplify=True)

    # Add per-edge speed_kph and travel_time (seconds)
    G = ox.add_edge_speeds(G)         # imputes speed_kph where missing
    G = ox.add_edge_travel_times(G)   # sets travel_time in seconds based on length and speed_kph

    return G


def _ring_lnglat_to_latlng(ring_lnglat: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Convert a Shapely ring coordinate sequence from (lng, lat) to (lat, lng),
    dropping the duplicated closing coordinate if present.
    """
    coords = list(ring_lnglat)
    if len(coords) >= 2 and coords[0] == coords[-1]:
        coords = coords[:-1]
    # shapely gives (x=lng, y=lat) -> h3 wants (lat, lng)
    return [(lat, lng) for (lng, lat) in coords]


def _shapely_polygon_to_latlngpoly(poly: Polygon) -> LatLngPoly:
    outer = _ring_lnglat_to_latlng(poly.exterior.coords)
    holes = [_ring_lnglat_to_latlng(ring.coords) for ring in poly.interiors]
    return LatLngPoly(outer, holes)


def _polygon_to_h3_cells(geom_wgs84: Union[Polygon, MultiPolygon], h3_res: int) -> List[str]:
    """
    Convert a Shapely Polygon or MultiPolygon (EPSG:4326) to H3 cells covering it.
    """
    if isinstance(geom_wgs84, Polygon):
        shape = _shapely_polygon_to_latlngpoly(geom_wgs84)
        return list(h3.h3shape_to_cells(shape, h3_res))

    if isinstance(geom_wgs84, MultiPolygon):
        polys = [_shapely_polygon_to_latlngpoly(p) for p in geom_wgs84.geoms]
        shape = LatLngMultiPoly(polys)
        return list(h3.h3shape_to_cells(shape, h3_res))

    raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom_wgs84)}")

def build_h3_grid_gdf(polygon_wgs84: Polygon, h3_res: int) -> gpd.GeoDataFrame:
    """
    Return a GeoDataFrame of H3 polygons covering the study area polygon.
    """
    cells = _polygon_to_h3_cells(polygon_wgs84, h3_res)

    polys: List[Polygon] = []
    for cell in cells:
        # h3.cell_to_boundary returns (lat, lng) pairs; shapely wants (x=lng, y=lat)
        boundary_latlng = h3.cell_to_boundary(cell)
        boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
        polys.append(Polygon(boundary_lnglat))

    gdf = gpd.GeoDataFrame({"h3_cell": cells, "geometry": polys}, crs="EPSG:4326")
    return gdf


def build_h3_travel_graph_from_osm(
    G_osm: nx.MultiDiGraph,
    h3_res: int,
    *,
    weight_attr: str = "travel_time",
    sample_miles: Optional[float] = None,
    sample_meters: Optional[float] = None,
    combine_parallel: str = "min",
    enforce_min_step_time: bool = True,
    v_max_mph: Optional[float] = None,
    v_max_kph: Optional[float] = None,
    floor_speed_source: str = "vmax",
    min_osm_speed_mph: Optional[float] = None,
    min_osm_speed_kph: Optional[float] = None,
    preserve_way_geometry: bool = True,
    way_cell_refine_max_depth: int = 18,
) -> nx.Graph:
    """
    Build an H3-level undirected travel graph by sampling along each OSM edge geometry.

    Key differences vs endpoint-only:
      - Samples points along each edge every sample_miles (or sample_meters) in a projected CRS
      - Converts sampled points to H3 cells
      - Connects successive cells, splitting edge travel_time across the steps

    combine_parallel:
      - "min" keeps the smallest travel_time for an H3 adjacency across multiple roads
      - "mean" averages (usually not recommended for fastest-path travel times)
      - "p25" uses the 25th percentile of observed travel_time candidates for an
        adjacency (faster-path bias while still using more than a single minimum)

    enforce_min_step_time (Variant A):
      - Applies a per-adjacency time floor based on H3 centroid distance:
        min_time = centroid_distance_m / speed_mps
        edge_time = max(observed_edge_time, min_time)

    floor_speed_source:
      - "vmax": speed_mps is derived from v_max_mph (or v_max_kph) (strict minimum-time floor behavior)
      - "osm_median": speed_mps is derived from the median OSM speed_kph observed for that
        adjacency, clamped to [min_osm_speed_mph, v_max_mph] (or the kph equivalents)
        (keeps OSM connectivity and introduces a local speed signal)

    preserve_way_geometry:
      - True: derive crossed H3 cells from the edge geometry itself using recursive midpoint
        refinement so transitions follow the way shape.
      - False: use sampled cells directly, with per-jump grid-path expansion fallback.

    way_cell_refine_max_depth:
      - maximum recursion depth for geometry-driven midpoint refinement when
        preserve_way_geometry=True.
    """
    sample_meters_value = _resolve_distance_meters(
        miles=sample_miles,
        meters=sample_meters,
        default_meters=100.0,
    )
    v_max_kph_value = _resolve_speed_kph(
        mph=v_max_mph,
        kph=v_max_kph,
        default_kph=60.0,
    )
    min_osm_speed_kph_value = _resolve_speed_kph(
        mph=min_osm_speed_mph,
        kph=min_osm_speed_kph,
        default_kph=10.0,
    )

    if combine_parallel not in {"min", "mean", "p25"}:
        raise ValueError("combine_parallel must be one of: 'min', 'mean', 'p25'")
    if floor_speed_source not in {"vmax", "osm_median"}:
        raise ValueError("floor_speed_source must be one of: 'vmax', 'osm_median'")
    if v_max_kph_value <= 0:
        raise ValueError("v_max_kph must be > 0")
    if min_osm_speed_kph_value <= 0:
        raise ValueError("min_osm_speed_kph must be > 0")

    # Project graph to a metric CRS so sampling is in meters
    Gp = ox.project_graph(G_osm)
    crs_proj = Gp.graph.get("crs")
    if crs_proj is None:
        raise ValueError("Projected graph is missing CRS.")

    to_wgs84 = pyproj.Transformer.from_crs(crs_proj, "EPSG:4326", always_xy=True)

    # We will build edges in an undirected graph with travel_time weights in seconds
    H = nx.Graph()

    def _edge_linestring(u: Any, v: Any, data: Dict[str, Any]) -> LineString:
        geom = data.get("geometry")
        if geom is not None and not geom.is_empty:
            return geom

        # fallback: straight line between node coordinates (in projected CRS)
        ux = Gp.nodes[u].get("x")
        uy = Gp.nodes[u].get("y")
        vx = Gp.nodes[v].get("x")
        vy = Gp.nodes[v].get("y")
        if ux is None or uy is None or vx is None or vy is None:
            raise ValueError("Missing node coordinates for edge geometry fallback.")
        return LineString([(ux, uy), (vx, vy)])

    def _h3_cell_from_xy(x: float, y: float) -> str:
        lng, lat = to_wgs84.transform(x, y)
        return h3.latlng_to_cell(lat, lng, h3_res)

    def _grid_distance(a: str, b: str) -> Optional[int]:
        try:
            return int(h3.grid_distance(a, b))
        except Exception:
            return None

    def _fallback_grid_path(a: str, b: str) -> List[str]:
        try:
            path = list(h3.grid_path_cells(a, b))
            if len(path) >= 2:
                return path
        except Exception:
            pass
        return [a, b]

    def _cells_between_distances(
        line: LineString,
        d0: float,
        d1: float,
        c0: str,
        c1: str,
        depth: int,
    ) -> List[str]:
        if c0 == c1:
            return [c0]

        gd = _grid_distance(c0, c1)
        if gd == 1:
            return [c0, c1]

        if depth >= int(way_cell_refine_max_depth):
            return _fallback_grid_path(c0, c1)

        mid = (float(d0) + float(d1)) / 2.0
        if not (d0 < mid < d1):
            return _fallback_grid_path(c0, c1)

        mid_pt = line.interpolate(mid)
        c_mid = _h3_cell_from_xy(float(mid_pt.x), float(mid_pt.y))

        left = _cells_between_distances(line, d0, mid, c0, c_mid, depth + 1)
        right = _cells_between_distances(line, mid, d1, c_mid, c1, depth + 1)
        if not left:
            return right
        if not right:
            return left
        if left[-1] == right[0]:
            return left + right[1:]
        return left + right

    def _trace_cells_along_line(line: LineString, dists: np.ndarray) -> List[str]:
        sampled: List[Tuple[float, str]] = []
        last_cell: Optional[str] = None
        for d in dists.tolist():
            pt = line.interpolate(float(d))
            c = _h3_cell_from_xy(float(pt.x), float(pt.y))
            if c != last_cell:
                sampled.append((float(d), c))
                last_cell = c

        if len(sampled) < 2:
            return [sampled[0][1]] if sampled else []

        cells: List[str] = [sampled[0][1]]
        for (d0, c0), (d1, c1) in zip(sampled[:-1], sampled[1:]):
            if c0 == c1:
                continue
            if preserve_way_geometry:
                seg_cells = _cells_between_distances(line, d0, d1, c0, c1, 0)
            else:
                seg_cells = _fallback_grid_path(c0, c1) if _grid_distance(c0, c1) not in {0, 1} else [c0, c1]
            if not seg_cells:
                continue
            if cells[-1] == seg_cells[0]:
                cells.extend(seg_cells[1:])
            else:
                cells.extend(seg_cells)

        # Consecutive dedupe for robustness after fallback stitching.
        deduped: List[str] = []
        prev: Optional[str] = None
        for c in cells:
            if c != prev:
                deduped.append(c)
                prev = c
        return deduped

    geod = pyproj.Geod(ellps="WGS84")

    def _centroid_distance_m(cell_a: str, cell_b: str) -> float:
        lat_a, lng_a = h3.cell_to_latlng(cell_a)
        lat_b, lng_b = h3.cell_to_latlng(cell_b)
        _, _, dist_m = geod.inv(lng_a, lat_a, lng_b, lat_b)
        return float(dist_m)

    # Temporary store for parallel edge combining
    # key: (a, b) sorted -> list of candidate weights (seconds)
    pair_to_weights: Dict[Tuple[str, str], List[float]] = {}
    pair_to_speeds_kph: Dict[Tuple[str, str], List[float]] = {}

    for u, v, k, data in Gp.edges(keys=True, data=True):
        w = data.get(weight_attr)
        if w is None:
            continue

        try:
            edge_time = float(w)
        except Exception:
            continue

        line = _edge_linestring(u, v, data)
        length_m = float(data.get("length") or line.length)
        if length_m <= 0:
            continue

        # Number of segments based on sampling resolution
        n_segs = int(np.ceil(length_m / float(sample_meters_value)))
        n_segs = max(n_segs, 1)

        # Sample along the line at segment boundaries
        dists = np.linspace(0.0, length_m, n_segs + 1)

        # Convert sampled points to H3 cells and refine transitions so crossed cells
        # follow way geometry instead of geometric shortcuts through hex space.
        cells = _trace_cells_along_line(line, dists)

        if len(cells) < 2:
            continue

        # Split edge_time across traced steps between successive cells.
        sampled_step_time = edge_time / float(len(cells) - 1)
        edge_speed_kph = _coerce_speed_to_kph(data.get("speed_kph"))

        for a, b in zip(cells[:-1], cells[1:]):
            if a == b:
                continue
            x, y = (a, b) if a < b else (b, a)
            key = (x, y)
            pair_to_weights.setdefault(key, []).append(sampled_step_time)
            if edge_speed_kph is not None:
                pair_to_speeds_kph.setdefault(key, []).append(edge_speed_kph)

    # Combine parallel weights into final graph edges
    for (a, b), ws in pair_to_weights.items():
        if combine_parallel == "min":
            w_out = float(np.min(ws))
        elif combine_parallel == "mean":
            w_out = float(np.mean(ws))
        else:
            w_out = float(np.percentile(ws, 25.0))
        observed_w = float(w_out)
        dist_m = _centroid_distance_m(a, b)
        osm_speed_median_kph: Optional[float] = None
        if pair_to_speeds_kph.get((a, b)):
            osm_speed_median_kph = float(np.median(pair_to_speeds_kph[(a, b)]))
        speed_kph_for_floor: Optional[float] = None
        min_step_time: Optional[float] = None
        floor_applied = False

        if enforce_min_step_time:
            # NOTE: This is where the logic to determine the minimum speed is set.
            speed_kph_for_floor = float(v_max_kph_value)
            if floor_speed_source == "osm_median":
                if osm_speed_median_kph is not None:
                    speed_kph_for_floor = float(osm_speed_median_kph)
                speed_kph_for_floor = max(
                    float(min_osm_speed_kph_value),
                    min(float(v_max_kph_value), speed_kph_for_floor),
                )
            speed_mps = speed_kph_for_floor / 3.6
            min_step_time = dist_m / speed_mps
            w_out = max(w_out, float(min_step_time))
            floor_applied = bool(w_out > observed_w + 1e-9)

        H.add_edge(
            a,
            b,
            **{
                weight_attr: float(w_out),
                "observed_step_time_raw_sec": observed_w,
                "centroid_dist_m": float(dist_m),
                "centroid_dist_miles": _meters_to_miles(dist_m),
                "osm_median_speed_kph": osm_speed_median_kph,
                "osm_median_speed_mph": _kph_to_mph(osm_speed_median_kph) if osm_speed_median_kph is not None else None,
                "floor_speed_kph": speed_kph_for_floor,
                "floor_speed_mph": _kph_to_mph(speed_kph_for_floor) if speed_kph_for_floor is not None else None,
                "min_step_time_sec": min_step_time,
                "floor_applied": floor_applied,
            },
        )

    return H


def _ensure_wgs84_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set gdf.crs before calling.")
    if str(gdf.crs).lower() in {"epsg:4326", "wgs84"}:
        return gdf
    return gdf.to_crs("EPSG:4326")


def snap_cell_to_graph(
    cell: str,
    graph_nodes: Set[str],
    *,
    max_k: int = 10,
) -> Optional[str]:
    """
    If cell is not in graph_nodes, expand k-rings (grid_disk) until a cell is found that is.
    Returns the snapped cell, or None if not found within max_k.
    Deterministic: searches k=0..max_k and returns the first found in that ring.
    """
    if cell in graph_nodes:
        return cell

    for k in range(1, max_k + 1):
        # disk includes all cells within distance k (including inner rings)
        # To keep deterministic preference for nearer cells, we filter to exactly the ring k.
        ring = h3.grid_ring(cell, k)
        # grid_ring can fail near pentagons; fall back to disk difference
        if not ring:
            disk = set(h3.grid_disk(cell, k))
            inner = set(h3.grid_disk(cell, k - 1))
            ring = list(disk - inner)

        # Deterministic ordering: sort cell ids lexicographically
        for cand in sorted(ring):
            if cand in graph_nodes:
                return cand

    return None


def h3_path_to_linestring(path_cells: List[str]) -> LineString:
    coords = []
    for c in path_cells:
        lat, lng = h3.cell_to_latlng(c)
        coords.append((lng, lat))  # shapely x,y
    return LineString(coords)


def print_h3_route_breakdown(
    h3_graph: nx.Graph,
    path_cells: List[str],
    *,
    weight_attr: str = "travel_time",
    max_steps: Optional[int] = None,
) -> None:
    """
    Print per-step diagnostics for an H3 route.
    Uses edge attributes added by build_h3_travel_graph_from_osm.
    """
    if len(path_cells) < 2:
        print("Route breakdown skipped: path has fewer than 2 cells.")
        return

    print("Route breakdown (per H3 step):")
    print(
        f"idx | from_cell -> to_cell {' '*len('2a8ac61dfffff')} | dist_mi | time_sec | implied_mph | raw_sec | floor_min_sec | floor_speed_mph | osm_median_mph | floor_applied"
    )

    total_time = 0.0
    total_dist_miles = 0.0
    n_steps = len(path_cells) - 1
    steps_to_print = n_steps if max_steps is None else min(n_steps, int(max_steps))

    for idx, (a, b) in enumerate(zip(path_cells[:-1], path_cells[1:]), start=1):
        if idx > steps_to_print:
            break
        ed = h3_graph[a][b]
        time_sec = float(ed.get(weight_attr, 0.0))
        dist_miles_raw = ed.get("centroid_dist_miles")
        if dist_miles_raw is not None:
            dist_miles = float(dist_miles_raw)
        else:
            dist_m = ed.get("centroid_dist_m")
            dist_miles = _meters_to_miles(float(dist_m)) if dist_m is not None else np.nan
        raw_sec = ed.get("observed_step_time_raw_sec")
        min_floor_sec = ed.get("min_step_time_sec")
        floor_speed_mph = ed.get("floor_speed_mph")
        if floor_speed_mph is None and ed.get("floor_speed_kph") is not None:
            floor_speed_mph = _kph_to_mph(float(ed["floor_speed_kph"]))
        osm_median_mph = ed.get("osm_median_speed_mph")
        if osm_median_mph is None and ed.get("osm_median_speed_kph") is not None:
            osm_median_mph = _kph_to_mph(float(ed["osm_median_speed_kph"]))
        floor_applied = bool(ed.get("floor_applied", False))
        implied_mph = (dist_miles / time_sec) * SECONDS_PER_HOUR if np.isfinite(dist_miles) and time_sec > 0 else np.nan

        total_time += time_sec
        if np.isfinite(dist_miles):
            total_dist_miles += dist_miles

        print(
            f"{idx:03d} | {a} -> {b} | "
            f"{dist_miles:8.3f} | {time_sec:8.1f} | {implied_mph:10.2f} | "
            f"{float(raw_sec) if raw_sec is not None else np.nan:7.1f} | "
            f"{float(min_floor_sec) if min_floor_sec is not None else np.nan:12.1f} | "
            f"{float(floor_speed_mph) if floor_speed_mph is not None else np.nan:15.1f} | "
            f"{float(osm_median_mph) if osm_median_mph is not None else np.nan:14.1f} | "
            f"{floor_applied}"
        )

    avg_mph = (total_dist_miles / total_time) * SECONDS_PER_HOUR if total_time > 0 else np.nan
    print(
        f"Route summary: steps={n_steps}, printed={steps_to_print}, "
        f"total_dist_miles={total_dist_miles:.3f}, total_time_sec={total_time:.1f}, avg_mph={avg_mph:.2f}"
    )
    if steps_to_print < n_steps:
        print(f"Note: truncated output. Increase max_steps to view all {n_steps} steps.")


def _path_distance_miles(h3_graph: nx.Graph, path_cells: Sequence[str]) -> float:
    total_miles = 0.0
    for a, b in zip(path_cells[:-1], path_cells[1:]):
        ed = h3_graph[a][b]
        dist_miles_raw = ed.get("centroid_dist_miles")
        if dist_miles_raw is not None:
            total_miles += float(dist_miles_raw)
            continue
        dist_m_raw = ed.get("centroid_dist_m")
        if dist_m_raw is not None and np.isfinite(float(dist_m_raw)):
            total_miles += _meters_to_miles(float(dist_m_raw))
    return float(total_miles)


def assign_nearest_target_by_h3_network(
    source_points: gpd.GeoDataFrame,
    target_points: gpd.GeoDataFrame,
    *,
    target_id_col: str,
    out_col: str = "nearest_target_id",
    h3_graph: nx.Graph,
    h3_res: int,
    weight_attr: str = "travel_time",
    tie_break_project_crs: str = "EPSG:3857",
    out_path_col: Optional[str] = "h3_path",
    out_time_col: Optional[str] = "h3_travel_time",
    out_distance_col: Optional[str] = "h3_travel_miles",
    debug_route_breakdown_source_idx: Optional[int] = 1,
    debug_route_breakdown_max_steps: Optional[int] = None,
) -> gpd.GeoDataFrame:
    """
    For each source point, find the closest target point by network travel time on the H3 graph,
    then write the chosen target_id_col into a new column on the source GeoDataFrame.

    Approach:
      1) Map all target points to H3 cells.
      2) Run multi-source Dijkstra on the H3 graph from all target cells at once.
         This gives, for every reachable cell, the travel time to the closest target cell,
         and also which target cell was the closest.
      3) Map each source point to an H3 cell and look up the closest target cell.
      4) If multiple target points fall in that cell, break ties by planar distance in a projected CRS.

    Returns a copy of source_points with out_col added.
    Time outputs are seconds. Distance outputs are miles.
    """
    if target_id_col not in target_points.columns:
        raise ValueError(f"target_id_col '{target_id_col}' not found in target_points")

    src = _ensure_wgs84_points(source_points).copy()
    tgt = _ensure_wgs84_points(target_points).copy()

    # Map target points -> cells
    tgt_cells: List[Optional[str]] = []
    for geom in tgt.geometry:
        if geom is None or geom.is_empty:
            tgt_cells.append(None)
            continue
        if not isinstance(geom, Point):
            raise ValueError("target_points geometry must be Point")
        tgt_cells.append(h3.latlng_to_cell(geom.y, geom.x, h3_res))

    tgt["h3_cell"] = tgt_cells
    tgt = tgt.dropna(subset=["h3_cell"]).copy()
    graph_nodes: set[str] = set(h3_graph.nodes)

    # Snap target cells into the graph
    tgt["h3_cell_graph"] = tgt["h3_cell"].apply(
        lambda c: snap_cell_to_graph(c, graph_nodes, max_k=10)
    )

    # Drop any targets that could not be snapped
    tgt = tgt.dropna(subset=["h3_cell_graph"]).copy()

    cell_to_target_idx: Dict[str, List[int]] = {}
    for idx, cell in zip(tgt.index.tolist(), tgt["h3_cell_graph"].tolist()):
        cell_to_target_idx.setdefault(cell, []).append(idx)

    if len(cell_to_target_idx) == 0:
        raise ValueError("No valid target points after mapping to H3 cells.")

    # Pre-Dijkstra diagnostics
    print("H3 graph nodes:", h3_graph.number_of_nodes())
    print("H3 graph edges:", h3_graph.number_of_edges())
    print("Unique target cells (snapped):", len(cell_to_target_idx))
    # How many target cells are actually nodes (should be all)
    missing_sources = [c for c in cell_to_target_idx.keys() if c not in h3_graph]
    print("Missing target source nodes (should be 0):", len(missing_sources))

    # Multi-source Dijkstra
    # distances[cell] = best travel time to nearest target cell
    # paths[cell] = path of cells from cell to the nearest target cell (includes both ends)
    distances, paths = nx.multi_source_dijkstra(
        h3_graph,
        sources=list(cell_to_target_idx.keys()),
        weight=weight_attr,
    )
    print("Dijkstra reached nodes:", len(distances))
    if len(distances) != h3_graph.number_of_nodes():
        warnings.warn("Warning: Dijkstra did not reach all nodes. Graph may be disconnected.")

    # Prepare projected versions for tie-breaking by Euclidean distance
    src_proj = src.to_crs(tie_break_project_crs)
    tgt_proj = tgt.to_crs(tie_break_project_crs)

    chosen_ids: List[Optional[Any]] = []

    for i, geom in enumerate(src.geometry):
        if geom is None or geom.is_empty:
            chosen_ids.append(None)
            continue
        if not isinstance(geom, Point):
            raise ValueError("source_points geometry must be Point")

        s_cell = h3.latlng_to_cell(geom.y, geom.x, h3_res)

        # Snap source cell into graph
        s_cell_graph = snap_cell_to_graph(s_cell, graph_nodes, max_k=10)

        # Diagnostics
        if i < 10:
            print(
                "src", i,
                "cell", s_cell,
                "snapped", s_cell_graph,
                "snapped_in_graph", (s_cell_graph in h3_graph) if s_cell_graph else None,
                "snapped_in_paths", (s_cell_graph in paths) if s_cell_graph else None,
            )

        if s_cell_graph is None or s_cell_graph not in paths:
            chosen_ids.append(None)
            continue

        # Test prints
        if i < 10:
            print("path length cells:", len(paths[s_cell_graph]))
            print("travel_time (seconds):", float(distances[s_cell_graph]))

        # Also get the path that was taken to get to the destination
        path_cells: list[str] = paths[s_cell_graph]
        travel_time = float(distances[s_cell_graph])
        travel_miles = _path_distance_miles(h3_graph, path_cells)
        if debug_route_breakdown_source_idx is not None and i == debug_route_breakdown_source_idx:
            print_h3_route_breakdown(
                h3_graph,
                path_cells,
                weight_attr=weight_attr,
                max_steps=debug_route_breakdown_max_steps,
            )
        if out_path_col is not None:
            # list is fine, but you can also do "|".join(path_cells)
            src.loc[src.index[i], out_path_col] = "|".join(path_cells)
        if out_time_col is not None:
            src.loc[src.index[i], out_time_col] = travel_time
        if out_distance_col is not None:
            src.loc[src.index[i], out_distance_col] = travel_miles

        # Nearest target cell is the first element in the returned path
        nearest_cell = paths[s_cell_graph][0]
        candidate_idxs = cell_to_target_idx.get(nearest_cell, [])
        if not candidate_idxs:
            chosen_ids.append(None)
            continue

        if len(candidate_idxs) == 1:
            chosen_ids.append(tgt.loc[candidate_idxs[0], target_id_col])
            continue

        # Tie-break: choose closest target point in that cell by planar distance
        s_geom_proj = src_proj.geometry.iloc[i]
        dists = []
        for tidx in candidate_idxs:
            d = s_geom_proj.distance(tgt_proj.loc[tidx].geometry)
            dists.append((float(d), tidx))
        dists.sort(key=lambda x: x[0])
        best_tidx = dists[0][1]
        chosen_ids.append(tgt.loc[best_tidx, target_id_col])

    src[out_col] = chosen_ids
    return src



def build_route_gdf_from_assignment(
    assigned_src: gpd.GeoDataFrame,
    *,
    src_id_col: str,
    path_col: str = "h3_path",
    time_col: str = "h3_travel_time",
    distance_col: str = "h3_travel_miles",
    target_id_col: str = "nearest_tgt_id",
) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []

    for _, r in assigned_src.iterrows():
        p = r.get(path_col)
        if p is None or (isinstance(p, float) and np.isnan(p)):
            continue
        if not isinstance(p, str) or not p.strip():
            continue

        cells = p.split("|")
        geom = h3_path_to_linestring(cells)

        rows.append(
            {
                src_id_col: r.get(src_id_col),
                target_id_col: r.get(target_id_col),
                time_col: r.get(time_col),
                distance_col: r.get(distance_col),
                "n_cells": len(cells),
                "geometry": geom,
            }
        )

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def explode_paths_to_h3_cells(
    assigned_src: gpd.GeoDataFrame,
    *,
    src_id_col: str,
    path_col: str = "h3_path",
) -> gpd.GeoDataFrame:
    rows: list[dict[str, Any]] = []
    for _, r in assigned_src.iterrows():
        p = r.get(path_col)
        if not isinstance(p, str) or not p.strip():
            continue
        for step, cell in enumerate(p.split("|")):
            rows.append({src_id_col: r.get(src_id_col), "step": step, "h3_cell": cell})
    return gpd.GeoDataFrame(rows)


"""
What you will probably want to improve next (but this is enough to start testing)

Edge sampling along geometry: Instead of mapping only OSM edge endpoints to cells, sample points along each edge geometry and connect the sequence of cells it crosses. That reduces “cell-skipping” artifacts at coarser resolutions.

Directed travel times: right now we collapse to an undirected H3 graph. If you want one-way and turn restrictions to matter, you can keep a directed H3 graph and run directed multi-source Dijkstra.

Resolution selection: for county-scale work, h3_res=8 is a decent first test; 9 is finer. You can parameterize this and compare runtime vs error.

If you paste in what your two point layers represent (providers vs population centroids, building points vs centroids, etc.) and what travel mode you care about (drive vs walk), I can tighten the graph-building step so it aligns with your modeling assumptions (especially speed defaults, one-way handling, and how you want to represent “entering” and “exiting” a hex cell).
"""
