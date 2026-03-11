from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List

import geopandas as gpd
import h3
import networkx as nx
from shapely.geometry import LineString, Point, Polygon


def _cell_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def _cell_centroid_point(cell: str) -> Point:
    lat, lng = h3.cell_to_latlng(cell)
    return Point(float(lng), float(lat))


def _load_graph(path: Path) -> nx.Graph:
    with path.open("rb") as f:
        graph = pickle.load(f)
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Pickle did not contain a NetworkX graph: {path}")
    return graph


def _graph_to_node_gdf(H: nx.Graph) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for cell in H.nodes():
        rows.append(
            {
                "h3_cell": str(cell),
                "degree": int(H.degree[cell]),
                "geometry": _cell_polygon(str(cell)),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _graph_to_centroid_gdf(H: nx.Graph) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for cell in H.nodes():
        rows.append(
            {
                "h3_cell": str(cell),
                "degree": int(H.degree[cell]),
                "geometry": _cell_centroid_point(str(cell)),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def _graph_to_edge_gdf(H: nx.Graph, *, weight_attr: str) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for a, b, d in H.edges(data=True):
        pa = _cell_centroid_point(str(a))
        pb = _cell_centroid_point(str(b))
        rows.append(
            {
                "h3_from": str(a),
                "h3_to": str(b),
                "travel_time": float(d.get(weight_attr, 0.0)),
                "obs_raw_sec": d.get("observed_step_time_raw_sec"),
                "min_step_sec": d.get("min_step_time_sec"),
                "floor_appl": bool(d.get("floor_applied", False)),
                "dist_mi": d.get("centroid_dist_miles"),
                "floor_mph": d.get("floor_speed_mph"),
                "osm_med_mph": d.get("osm_median_speed_mph"),
                "geometry": LineString([pa, pb]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def export_h3_graph_pickle_to_gpkg(
    *,
    pkl_path: str,
    out_gpkg_path: str,
    weight_attr: str = "travel_time",
) -> Dict[str, Any]:
    pkl = Path(pkl_path).resolve()
    out = Path(out_gpkg_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    H = _load_graph(pkl)
    nodes_poly = _graph_to_node_gdf(H)
    nodes_cent = _graph_to_centroid_gdf(H)
    edges = _graph_to_edge_gdf(H, weight_attr=weight_attr)

    nodes_poly.to_file(out, layer="h3_nodes_poly", driver="GPKG")
    nodes_cent.to_file(out, layer="h3_nodes_centroid", driver="GPKG", mode="a")
    edges.to_file(out, layer="h3_edges_centroid_line", driver="GPKG", mode="a")

    return {
        "pkl_path": str(pkl),
        "out_gpkg_path": str(out),
        "n_nodes": int(H.number_of_nodes()),
        "n_edges": int(H.number_of_edges()),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert H3 NetworkX pickle graph to GPKG for QGIS.")
    p.add_argument("--pkl", required=True, help="Path to .pkl H3 graph.")
    p.add_argument("--out-gpkg", required=True, help="Output .gpkg path.")
    p.add_argument("--weight-attr", default="travel_time", help="Edge weight attribute.")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = export_h3_graph_pickle_to_gpkg(
        pkl_path=str(args.pkl),
        out_gpkg_path=str(args.out_gpkg),
        weight_attr=str(args.weight_attr),
    )
    print("Export complete.")
    print("PKL:", result["pkl_path"])
    print("GPKG:", result["out_gpkg_path"])
    print("Nodes:", result["n_nodes"])
    print("Edges:", result["n_edges"])
