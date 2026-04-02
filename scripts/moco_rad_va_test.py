from __future__ import annotations

import os.path
from dataclasses import dataclass
from typing import Tuple

import networkx as nx
import numpy as np
import geopandas as gpd
import osmnx as ox
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from ccaoa import core as cf, raccoon as rc, gis

try:
    from _bootstrap import ensure_src_on_path
except ImportError:
    from scripts._bootstrap import ensure_src_on_path

ensure_src_on_path()

from osmh3nx import calibrate as calx
from osmh3nx import network_h3 as hnetx
from osmh3nx.io import write_layers_to_gpkg

@dataclass(frozen=True)
class StudyArea:
    name: str
    polygon_wgs84: Polygon  # EPSG:4326


def get_study_area_radford_montgomery(output_osm_test: bool = False) -> StudyArea:
    """
    Fetch admin boundaries for Radford City, VA and Montgomery County, VA, union them,
    and return as a single polygon (WGS84).
    """
    # OSMnx geocoding returns a GeoDataFrame in EPSG:4326
    gdf_radford = ox.geocode_to_gdf("Radford, Virginia, USA")
    gdf_monty = ox.geocode_to_gdf("Montgomery County, Virginia, USA")
    if cf.string_to_bool(output_osm_test):
        gis.gdf_to_file(gdf_radford, os.path.join(os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing"),"radford_osm.geojson"), overwrite=True)
        gis.gdf_to_file(gdf_monty, os.path.join(os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing"),"moco_osm.geojson"), overwrite=True)

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


def demo_radford_montgomery_pipeline(
    *,
    h3_res: int = 8,
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME,
    test_outs: bool = False
) -> Tuple[StudyArea, nx.MultiDiGraph, nx.Graph, gpd.GeoDataFrame]:
    """
    Convenience function to build everything for the Radford + Montgomery test case.
    """
    area = get_study_area_radford_montgomery(output_osm_test=test_outs)
    G_osm = hnetx.download_osm_graph_drive(area.polygon_wgs84)
    H_h3, _profile = calx.build_calibrated_h3_graph_from_osm(
        G_osm,
        h3_res=h3_res,
        profile_name=calibration_profile_name,
    )
    # TODO delete test prints; used for debugging
    print("H_h3 nodes:", H_h3.number_of_nodes())
    print("H_h3 edges:", H_h3.number_of_edges())
    print("edge travel_time min/median/max:",
          np.min([d["travel_time"] for _, _, d in H_h3.edges(data=True)]),
          np.median([d["travel_time"] for _, _, d in H_h3.edges(data=True)]),
          np.max([d["travel_time"] for _, _, d in H_h3.edges(data=True)]))
    #
    h3_grid: gpd.GeoDataFrame = hnetx.build_h3_grid_gdf(area.polygon_wgs84, h3_res=h3_res)
    return area, G_osm, H_h3, h3_grid



# Example usage (replace these with your real point layers):
if __name__ == "__main__":
    import time
    start = time.time()

    ox.settings.use_cache = True
    ox.settings.log_console = True
    resolution_h3_cell: int = 10

    output_tests: bool = False
    area, G_osm, H_h3, h3_grid = demo_radford_montgomery_pipeline(h3_res=resolution_h3_cell, test_outs=output_tests)  # Returns StudyArea, nx.MultiDiGraph, nx.Graph, gpd.GeoDataFrame
    query_weight_attr = str(
        H_h3.graph.get(
            "default_query_weight_attr",
            calx.get_default_query_weight_attr(calx.get_calibration_profile()),
        )
    )

    print(area)
    print(G_osm)
    print(H_h3)
    # print(h3_grid)
    if output_tests:
        gis.gdf_to_file(h3_grid,os.path.join(os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing"),f"mocorad_h3_grid_rez{str(resolution_h3_cell)}.geojson"), overwrite=True)
    print()
    if H_h3.is_directed():
        weak_component_count = nx.number_weakly_connected_components(H_h3)
        weak_sizes = sorted((len(c) for c in nx.weakly_connected_components(H_h3)), reverse=True)
        strong_component_count = nx.number_strongly_connected_components(H_h3)
        strong_sizes = sorted((len(c) for c in nx.strongly_connected_components(H_h3)), reverse=True)
        print("Weakly connected components:", weak_component_count)
        print("Largest weak components:", weak_sizes[:10])
        print("Strongly connected components:", strong_component_count)
        print("Largest strong components:", strong_sizes[:10])
    else:
        print("Connected components:", nx.number_connected_components(H_h3))
        sizes = sorted((len(c) for c in nx.connected_components(H_h3)), reverse=True)
        print("Largest components:", sizes[:10])
    # Synthetic example points
    src_points = gpd.GeoDataFrame(
        {"src_id": ["a", "b"]},
        geometry=[Point(-80.54795104042991, 37.1298928283625), Point(-80.40, 37.24)],
        crs="EPSG:4326",
    )

    tgt_points = gpd.GeoDataFrame(
        {"tgt_id": [101, 102, 103]},
        geometry=[Point(-80.57697081226758, 37.13176833876923), Point(-80.43, 37.21), Point(-80.32, 37.28)],
        crs="EPSG:4326",
    )

    pathcol = "h3_path"
    out: gpd.GeoDataFrame = hnetx.assign_nearest_target_by_h3_network(
        src_points,
        tgt_points,
        target_id_col="tgt_id",
        out_col="nearest_tgt_id",
        h3_graph=H_h3,
        h3_res=int(resolution_h3_cell),
        weight_attr=query_weight_attr,
        out_path_col=pathcol
    )

    # Try to visualize the path taken here
    distinct_paths = out[pathcol].dropna().unique().tolist()
    for dp in distinct_paths:
        cells = dp.split("|")
        line = hnetx.h3_path_to_linestring(cells)
        print(line)

    report_weight_attr = str(
        H_h3.graph.get(
            "report_weight_attr",
            calx.get_default_report_weight_attr(calx.get_calibration_profile()),
        )
    )
    out["h3_travel_time_postcalibrated"] = out[pathcol].apply(
        lambda p: hnetx.path_weight_sum(H_h3, p.split("|"), weight_attr=report_weight_attr)
        if isinstance(p, str) and p.strip()
        else np.nan
    )

    # Output the routes to GDF
    routes = hnetx.build_route_gdf_from_assignment(
        out,
        src_id_col="src_id",
        path_col="h3_path",
        time_col="h3_travel_time",
        target_id_col="nearest_tgt_id",
    )
    # Get the actual h3 cells used for route
    path_cells = hnetx.explode_paths_to_h3_cells(out, src_id_col="src_id", path_col="h3_path")
    path_hexes = path_cells.merge(h3_grid, on="h3_cell", how="left")
    path_hexes = gpd.GeoDataFrame(path_hexes, geometry="geometry", crs=h3_grid.crs)

    export = True
    if export:
        # XPRT
        runtry = 9
        output_gpkg = os.path.join(
            os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing"),
            f"moco_radford_test_{str(runtry)}_rez{str(resolution_h3_cell)}.gpkg",
        )
        written_layers = write_layers_to_gpkg(
            output_gpkg,
            layers=[
                (f"h3_routes_{str(runtry)}_rez{str(resolution_h3_cell)}", routes),
                (f"h3_path_hexes_{str(runtry)}_rez{str(resolution_h3_cell)}", path_hexes),
                (f"outtestresult_{str(runtry)}_rez{str(resolution_h3_cell)}", out),
            ],
        )
        print("Wrote GPKG:", output_gpkg)
        print("Layers written:", written_layers)

    print(out[["src_id", "nearest_tgt_id", "h3_travel_miles", "h3_travel_time", "h3_travel_time_postcalibrated"]])

    cf.runtime(start)
    # return out
