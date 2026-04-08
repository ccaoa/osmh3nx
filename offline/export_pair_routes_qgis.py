from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence

import geopandas as gpd
import h3
import networkx as nx
import osmnx as ox
import pyproj
from shapely import wkt
from shapely.geometry import LineString, MultiLineString, Point, Polygon
from shapely.ops import linemerge, unary_union

# Ensure imports work when this script lives under ./offline.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import network_h3 as hnetx


def _parse_resolutions(text: str) -> List[int]:
    vals = [int(v.strip()) for v in text.split(",") if v.strip()]
    if not vals:
        raise ValueError("Provide at least one H3 resolution.")
    return vals


def _cell_to_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def _safe_linestring_from_segments(
    segments: Sequence[LineString],
) -> LineString | MultiLineString:
    merged = linemerge(unary_union(list(segments)))
    if isinstance(merged, (LineString, MultiLineString)):
        return merged
    # Fallback: build a simple chain if merge result is unexpected.
    coords: List[tuple[float, float]] = []
    for seg in segments:
        seg_coords = list(seg.coords)
        if not coords:
            coords.extend(seg_coords)
            continue
        if coords[-1] == seg_coords[0]:
            coords.extend(seg_coords[1:])
        else:
            coords.extend(seg_coords)
    return LineString(coords)


def _projected_nearest_node(
    G_proj: nx.MultiDiGraph,
    pt_wgs84: Point,
) -> Any:
    crs_proj = G_proj.graph.get("crs")
    if crs_proj is None:
        raise ValueError("Projected graph missing CRS.")
    tx = pyproj.Transformer.from_crs("EPSG:4326", crs_proj, always_xy=True)
    x, y = tx.transform(float(pt_wgs84.x), float(pt_wgs84.y))
    return ox.distance.nearest_nodes(G_proj, X=float(x), Y=float(y))


def _edge_linestring_from_data(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    data: Dict[str, Any],
) -> LineString:
    geom = data.get("geometry")
    if isinstance(geom, LineString) and not geom.is_empty:
        return geom
    if isinstance(geom, str):
        parsed = wkt.loads(geom)
        if isinstance(parsed, LineString) and not parsed.is_empty:
            return parsed

    ux = G.nodes[u].get("x")
    uy = G.nodes[u].get("y")
    vx = G.nodes[v].get("x")
    vy = G.nodes[v].get("y")
    if ux is None or uy is None or vx is None or vy is None:
        raise ValueError(
            f"Missing node coordinates for fallback geometry between {u} and {v}."
        )
    return LineString([(float(ux), float(uy)), (float(vx), float(vy))])


def _best_parallel_edge_data(
    G: nx.MultiDiGraph,
    u: Any,
    v: Any,
    *,
    weight_attr: str,
) -> Dict[str, Any]:
    edge_lookup = G.get_edge_data(u, v)
    if not edge_lookup:
        raise ValueError(f"No edge data found between nodes {u} and {v}.")

    best_data: Dict[str, Any] | None = None
    best_weight = float("inf")
    for _, data in edge_lookup.items():
        try:
            w = float(data.get(weight_attr, float("inf")))
        except Exception:
            w = float("inf")
        if w < best_weight:
            best_weight = w
            best_data = data

    if best_data is None:
        first_key = sorted(edge_lookup.keys(), key=lambda x: str(x))[0]
        best_data = edge_lookup[first_key]
    return dict(best_data)


def _osm_route_geometry_and_metrics(
    G_proj: nx.MultiDiGraph,
    route_nodes: Sequence[Any],
    *,
    weight_attr: str = "travel_time",
) -> tuple[LineString | MultiLineString, float, float]:
    if len(route_nodes) < 2:
        raise ValueError("Route must contain at least two nodes.")

    lines: List[LineString] = []
    travel_time_sec = 0.0
    length_m = 0.0
    for u, v in zip(route_nodes[:-1], route_nodes[1:]):
        data = _best_parallel_edge_data(G_proj, u, v, weight_attr=weight_attr)
        line = _edge_linestring_from_data(G_proj, u, v, data)
        lines.append(line)

        try:
            travel_time_sec += float(data.get(weight_attr, 0.0))
        except Exception:
            pass
        try:
            length_m += float(data.get("length", 0.0))
        except Exception:
            pass

    return (
        _safe_linestring_from_segments(lines),
        float(travel_time_sec),
        float(length_m),
    )


def _write_layer(
    gdf: gpd.GeoDataFrame,
    gpkg_path: Path,
    layer_name: str,
    *,
    append: bool,
) -> None:
    mode = "a" if append else "w"
    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode)


def export_pair_routes_to_qgis(
    *,
    pair_dir: Path,
    out_gpkg: Path,
    h3_resolutions: Iterable[int],
    weight_attr: str = "travel_time",
    snap_k: int = 10,
) -> Dict[str, Any]:
    manifest_path = pair_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    origin = wkt.loads(manifest["origin_wkt"])
    destination = wkt.loads(manifest["destination_wkt"])
    if not isinstance(origin, Point) or not isinstance(destination, Point):
        raise ValueError("origin_wkt and destination_wkt must be POINT geometries.")

    graphml_path = Path(manifest["osm_graphml_path"])
    if not graphml_path.exists():
        raise FileNotFoundError(f"osm_drive.graphml not found: {graphml_path}")

    G = ox.load_graphml(str(graphml_path))
    Gp = ox.project_graph(G)

    origin_node = _projected_nearest_node(Gp, origin)
    destination_node = _projected_nearest_node(Gp, destination)
    osm_route_nodes = nx.shortest_path(
        Gp,
        source=origin_node,
        target=destination_node,
        weight=weight_attr,
    )
    osm_geom, osm_time_sec, osm_length_m = _osm_route_geometry_and_metrics(
        Gp,
        osm_route_nodes,
        weight_attr=weight_attr,
    )

    if out_gpkg.exists():
        out_gpkg.unlink()
    out_gpkg.parent.mkdir(parents=True, exist_ok=True)

    od_points = gpd.GeoDataFrame(
        [
            {"point_role": "origin", "geometry": origin},
            {"point_role": "destination", "geometry": destination},
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    _write_layer(od_points, out_gpkg, "od_points", append=False)

    osm_route_gdf = gpd.GeoDataFrame(
        [
            {
                "route_type": "osm_truth",
                "travel_time_sec": float(osm_time_sec),
                "travel_miles": float(osm_length_m / hnetx.METERS_PER_MILE),
                "n_route_nodes": int(len(osm_route_nodes)),
                "geometry": gpd.GeoSeries([osm_geom], crs=Gp.graph["crs"])
                .to_crs("EPSG:4326")
                .iloc[0],
            }
        ],
        geometry="geometry",
        crs="EPSG:4326",
    )
    _write_layer(osm_route_gdf, out_gpkg, "osm_route", append=True)

    appended = True
    h3_summary: List[Dict[str, Any]] = []
    for res in h3_resolutions:
        pkl_path = pair_dir / f"h3_drive_res{int(res)}.pkl"
        if not pkl_path.exists():
            h3_summary.append(
                {"h3_res": int(res), "status": "missing_pkl", "path": str(pkl_path)}
            )
            continue

        with pkl_path.open("rb") as f:
            H = pickle.load(f)
        if not isinstance(H, nx.Graph):
            h3_summary.append(
                {"h3_res": int(res), "status": "bad_pickle_type", "path": str(pkl_path)}
            )
            continue

        oc = h3.latlng_to_cell(float(origin.y), float(origin.x), int(res))
        dc = h3.latlng_to_cell(float(destination.y), float(destination.x), int(res))
        os_cell = hnetx.snap_cell_to_graph(oc, set(H.nodes), max_k=int(snap_k))
        ds_cell = hnetx.snap_cell_to_graph(dc, set(H.nodes), max_k=int(snap_k))
        if os_cell is None or ds_cell is None:
            h3_summary.append(
                {"h3_res": int(res), "status": "snap_failed", "path": str(pkl_path)}
            )
            continue

        path_cells = nx.shortest_path(
            H, source=os_cell, target=ds_cell, weight=weight_attr
        )
        travel_time_sec = float(
            nx.shortest_path_length(
                H, source=os_cell, target=ds_cell, weight=weight_attr
            )
        )
        travel_miles = float(
            sum(
                float(H[a][b].get("centroid_dist_miles", 0.0))
                for a, b in zip(path_cells[:-1], path_cells[1:])
            )
        )

        route_line = hnetx.h3_path_to_linestring(path_cells)
        route_gdf = gpd.GeoDataFrame(
            [
                {
                    "h3_res": int(res),
                    "travel_time_sec": travel_time_sec,
                    "travel_miles": travel_miles,
                    "n_cells": int(len(path_cells)),
                    "origin_cell": oc,
                    "destination_cell": dc,
                    "origin_cell_snap": os_cell,
                    "destination_cell_snap": ds_cell,
                    "geometry": route_line,
                }
            ],
            geometry="geometry",
            crs="EPSG:4326",
        )
        _write_layer(route_gdf, out_gpkg, f"h3_route_res{int(res)}", append=appended)
        appended = True

        hex_rows: List[Dict[str, Any]] = []
        for idx, cell in enumerate(path_cells):
            hex_rows.append(
                {
                    "h3_res": int(res),
                    "step": int(idx),
                    "h3_cell": str(cell),
                    "geometry": _cell_to_polygon(str(cell)),
                }
            )
        hex_gdf = gpd.GeoDataFrame(hex_rows, geometry="geometry", crs="EPSG:4326")
        _write_layer(hex_gdf, out_gpkg, f"h3_route_hexes_res{int(res)}", append=True)

        h3_summary.append(
            {
                "h3_res": int(res),
                "status": "ok",
                "path": str(pkl_path),
                "travel_time_sec": travel_time_sec,
                "travel_miles": travel_miles,
                "n_cells": int(len(path_cells)),
            }
        )

    return {
        "out_gpkg": str(out_gpkg),
        "pair_dir": str(pair_dir),
        "h3_results": h3_summary,
        "osm_travel_time_sec": float(osm_time_sec),
        "osm_travel_miles": float(osm_length_m / hnetx.METERS_PER_MILE),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export one pair's OSM and H3 routes to a QGIS-ready GeoPackage."
    )
    p.add_argument(
        "--pair-dir",
        required=True,
        help="Directory containing pair manifest and h3_drive_res*.pkl files.",
    )
    p.add_argument("--out-gpkg", required=True, help="Output GeoPackage path.")
    p.add_argument(
        "--resolutions", default="8", help="Comma-separated H3 resolutions to export."
    )
    p.add_argument(
        "--weight-attr",
        default="travel_time",
        help="Edge weight attribute for shortest path.",
    )
    p.add_argument(
        "--snap-k", type=int, default=10, help="Max k-ring snap distance for H3 cells."
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = export_pair_routes_to_qgis(
        pair_dir=Path(args.pair_dir).resolve(),
        out_gpkg=Path(args.out_gpkg).resolve(),
        h3_resolutions=_parse_resolutions(args.resolutions),
        weight_attr=str(args.weight_attr),
        snap_k=int(args.snap_k),
    )
    print("Route export complete.")
    print("GPKG:", result["out_gpkg"])
    print("OSM travel time sec:", round(result["osm_travel_time_sec"], 1))
    print("OSM travel miles:", round(result["osm_travel_miles"], 2))
    for row in result["h3_results"]:
        print(row)
