"""
I have the most recent versions of the python packages h3, networkx, and osmnx downloaded. I want to use OSM roads to do network analysis by using h3 hexagons at different scales. This would help me not have to tax the computational load nearly as much as doing a purely route-based network analysis with OSM roads directly.

This is my inspiration. Please look at this link for an idea. I want to build something similar in python (the back-end processing, not necessarily the interactive map in a Jupyter type space, cool though that is).
https://observablehq.com/@nrabinowitz/h3-travel-times

I want to look at a test case in Radford City and Montgomery County, Virginia to help begin to build this. Write me initial code that pulls OSM street data, h3 polygons at a county-appropriate scale, and a function that accepts two point geopandas datasets. The end goal is to see which point from one dataset is closest to the point in the other dataset by h3 network analysis. The unique ID of the compare dataset's closest point should be written to a new column in points from the source dataset.
"""

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from shapely.geometry import Point, Polygon, mapping
from shapely.ops import unary_union
import h3

from ccaoa import core as cf, raccoon as rc, gis

@dataclass(frozen=True)
class StudyArea:
    name: str
    polygon_wgs84: Polygon  # EPSG:4326


def get_study_area_radford_montgomery() -> StudyArea:
    """
    Fetch admin boundaries for Radford City, VA and Montgomery County, VA, union them,
    and return as a single polygon (WGS84).
    """
    # OSMnx geocoding returns a GeoDataFrame in EPSG:4326
    gdf_radford = ox.geocode_to_gdf("Radford, Virginia, USA")
    gdf_monty = ox.geocode_to_gdf("Montgomery County, Virginia, USA")

    geom = unary_union([gdf_radford.geometry.iloc[0], gdf_monty.geometry.iloc[0]])
    if geom.geom_type == "MultiPolygon":
        # union might still produce MultiPolygon; dissolve to a single polygon via unary_union
        geom = unary_union(list(geom.geoms))

    if geom.geom_type != "Polygon":
        raise ValueError(f"Unexpected geometry type after union: {geom.geom_type}")

    return StudyArea(
        name="Radford City + Montgomery County, VA",
        polygon_wgs84=geom,
    )


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


def _polygon_to_h3_cells(polygon_wgs84: Polygon, h3_res: int) -> List[str]:
    """
    Convert a shapely Polygon (EPSG:4326) to a list of H3 cell IDs covering it.
    """
    # h3.polygon_to_cells expects a GeoJSON-like dict with coordinates in lon/lat order
    geojson = mapping(polygon_wgs84)
    cells = list(h3.polygon_to_cells(geojson, h3_res))
    return cells


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
    agg: str = "min",
) -> nx.Graph:
    """
    Collapse an OSMnx street graph into an H3-level undirected graph where nodes are H3 cells and
    edges connect cells that have at least one OSM edge between them.

    Edge weight is an aggregation (min/sum/mean) of OSM edge weights between those two cells.

    This is an approximation:
      - Each OSM node is mapped to an H3 cell at the given resolution.
      - Each OSM edge becomes a connection between the origin cell and destination cell.
      - Multiple road segments between the same pair of cells are aggregated.

    Returns an undirected graph, appropriate for travel time computations.
    """
    if agg not in {"min", "sum", "mean"}:
        raise ValueError("agg must be one of: 'min', 'sum', 'mean'")

    # Map OSM node -> H3 cell
    node_to_cell: Dict[Any, str] = {}
    for n, data in G_osm.nodes(data=True):
        lat = data.get("y", None)
        lng = data.get("x", None)
        if lat is None or lng is None:
            continue
        node_to_cell[n] = h3.latlng_to_cell(lat, lng, h3_res)

    # Collect weights between cell pairs
    # Use sorted tuple (a, b) so we can build an undirected representation
    cellpair_to_weights: Dict[Tuple[str, str], List[float]] = {}

    for u, v, k, data in G_osm.edges(keys=True, data=True):
        cu = node_to_cell.get(u)
        cv = node_to_cell.get(v)
        if cu is None or cv is None or cu == cv:
            continue

        w = data.get(weight_attr)
        if w is None:
            continue

        try:
            w_float = float(w)
        except Exception:
            continue

        a, b = (cu, cv) if cu < cv else (cv, cu)
        cellpair_to_weights.setdefault((a, b), []).append(w_float)

    # Build H3 graph
    H = nx.Graph()
    for (a, b), weights in cellpair_to_weights.items():
        if agg == "min":
            w_out = float(np.min(weights))
        elif agg == "sum":
            w_out = float(np.sum(weights))
        else:
            w_out = float(np.mean(weights))

        H.add_edge(a, b, **{weight_attr: w_out})

    return H


def _ensure_wgs84_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS. Set gdf.crs before calling.")
    if str(gdf.crs).lower() in {"epsg:4326", "wgs84"}:
        return gdf
    return gdf.to_crs("EPSG:4326")


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

    # Build dict: cell -> list of target rows (indices)
    cell_to_target_idx: Dict[str, List[int]] = {}
    for idx, cell in zip(tgt.index.tolist(), tgt["h3_cell"].tolist()):
        cell_to_target_idx.setdefault(cell, []).append(idx)

    if len(cell_to_target_idx) == 0:
        raise ValueError("No valid target points after mapping to H3 cells.")

    # Multi-source Dijkstra
    # distances[cell] = best travel time to nearest target cell
    # paths[cell] = path of cells from cell to the nearest target cell (includes both ends)
    distances, paths = nx.multi_source_dijkstra(
        h3_graph,
        sources=list(cell_to_target_idx.keys()),
        weight=weight_attr,
    )

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

        # If cell not reachable in H3 graph, no assignment
        if s_cell not in paths:
            chosen_ids.append(None)
            continue

        # Nearest target cell is the last element in the returned path
        nearest_cell = paths[s_cell][-1]
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


def demo_radford_montgomery_pipeline(
    *,
    h3_res: int = 8,
) -> Tuple[StudyArea, nx.MultiDiGraph, nx.Graph, gpd.GeoDataFrame]:
    """
    Convenience function to build everything for the Radford + Montgomery test case.
    """
    area = get_study_area_radford_montgomery()
    G_osm = download_osm_graph_drive(area.polygon_wgs84)
    H_h3 = build_h3_travel_graph_from_osm(G_osm, h3_res=h3_res, weight_attr="travel_time", agg="min")
    h3_grid = build_h3_grid_gdf(area.polygon_wgs84, h3_res=h3_res)
    return area, G_osm, H_h3, h3_grid


# Example usage (replace these with your real point layers):
if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.log_console = True

    area, G_osm, H_h3, h3_grid = demo_radford_montgomery_pipeline(h3_res=8)

    # Synthetic example points
    src_points = gpd.GeoDataFrame(
        {"src_id": ["a", "b"]},
        geometry=[Point(-80.57, 37.13), Point(-80.40, 37.24)],
        crs="EPSG:4326",
    )

    tgt_points = gpd.GeoDataFrame(
        {"tgt_id": [101, 102, 103]},
        geometry=[Point(-80.58, 37.13), Point(-80.43, 37.21), Point(-80.32, 37.28)],
        crs="EPSG:4326",
    )

    out = assign_nearest_target_by_h3_network(
        src_points,
        tgt_points,
        target_id_col="tgt_id",
        out_col="nearest_tgt_id",
        h3_graph=H_h3,
        h3_res=8,
        weight_attr="travel_time",
    )

    print(out[["src_id", "nearest_tgt_id"]])

"""
What you will probably want to improve next (but this is enough to start testing)

Edge sampling along geometry: Instead of mapping only OSM edge endpoints to cells, sample points along each edge geometry and connect the sequence of cells it crosses. That reduces “cell-skipping” artifacts at coarser resolutions.

Directed travel times: right now we collapse to an undirected H3 graph. If you want one-way and turn restrictions to matter, you can keep a directed H3 graph and run directed multi-source Dijkstra.

Resolution selection: for county-scale work, h3_res=8 is a decent first test; 9 is finer. You can parameterize this and compare runtime vs error.

If you paste in what your two point layers represent (providers vs population centroids, building points vs centroids, etc.) and what travel mode you care about (drive vs walk), I can tighten the graph-building step so it aligns with your modeling assumptions (especially speed defaults, one-way handling, and how you want to represent “entering” and “exiting” a hex cell).
"""
