from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import geopandas as gpd
import pandas as pd

import driveshed as dshed
import h3_osm_calibration as hcal


@dataclass(frozen=True)
class DriveshedTestConfig:
    requested_cities: Sequence[str] = ("Christiansburg", "Knoxville", "NYC", "Huntsville", "Pasadena")
    h3_res: int = 10
    upsampled_target_resolutions: Sequence[int] = (9, 8)
    max_travel_minutes: float = 15.0
    weight_attr: str = dshed.DEFAULT_H3_WEIGHT_ATTR
    search_buffer_speed_mph: float = 60.0
    search_buffer_factor: float = 1.4
    search_min_buffer_miles: float = 2.0
    sample_miles: float = 0.1
    combine_parallel: str = "min"
    directional: bool = True
    enforce_min_step_time: bool = True
    v_max_mph: float = 50.0
    floor_speed_source: str = "vmax"
    min_osm_speed_mph: float = 10.0 / dshed.network_h3.KM_PER_MILE
    route_weight_attr: str = "travel_time_route"
    route_floor_penalty_weight: float = 0.35
    report_weight_attr: str = "travel_time_postcalibrated"
    report_floor_penalty_weight: float = 1.0
    snap_max_k: int = 10
    osm_cache_dir: str | None = "cache"
    osm_force_refresh: bool = False


def _city_aliases() -> Dict[str, Sequence[str]]:
    return {
        "christiansburg": ("Christiansburg",),
        "knoxville": ("Knoxville",),
        "nyc": ("NYC", "New York City", "Manhattan"),
        "huntsville": ("Huntsville",),
        "pasadena": ("Pasadena",),
    }


def _resolve_city_rows(od_pairs: pd.DataFrame, requested_cities: Sequence[str]) -> pd.DataFrame:
    rows: List[pd.Series] = []
    aliases = _city_aliases()

    for requested in requested_cities:
        key = str(requested).strip().lower()
        city_names = aliases.get(key, (requested,))
        match = od_pairs[od_pairs["city"].isin(city_names)].copy()
        if match.empty:
            raise ValueError(f"No calibration CSV row found for requested city '{requested}'.")

        match = match.sort_values(["pair_id", "count"]).reset_index(drop=True)
        chosen = match.iloc[0].copy()
        chosen["requested_city"] = requested
        chosen["resolved_city"] = chosen["city"]
        rows.append(chosen)

    return pd.DataFrame(rows).reset_index(drop=True)


def _append_origin_metadata(
    gdf: gpd.GeoDataFrame,
    *,
    row: pd.Series,
    result: dshed.DriveshedResult,
) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["pair_id"] = int(row["pair_id"])
    out["count"] = int(row["count"])
    out["requested_city"] = row["requested_city"]
    out["city"] = row["city"]
    out["resolved_city"] = row["resolved_city"]
    out["state"] = row["state"]
    out["category"] = row["category"]
    out["origin_wkt"] = row["origin"]
    out["destination_wkt"] = row["destination"]
    out["h3_res"] = int(result.h3_res)
    out["max_travel_minutes"] = float(result.max_travel_minutes)
    out["weight_attr"] = result.weight_attr
    out["origin_h3_cell"] = result.origin_h3_cell
    out["origin_h3_cell_graph"] = result.origin_h3_cell_graph
    return out


def run_driveshed_test(
    *,
    csv_path: str,
    output_gpkg_path: str,
    config: DriveshedTestConfig,
) -> Dict[str, Any]:
    od_pairs = hcal.load_od_pairs(csv_path)
    selected = _resolve_city_rows(od_pairs, config.requested_cities)

    origin_rows: List[gpd.GeoDataFrame] = []
    search_rows: List[gpd.GeoDataFrame] = []
    polygon_rows: List[gpd.GeoDataFrame] = []
    cell_rows: List[gpd.GeoDataFrame] = []
    edge_rows: List[gpd.GeoDataFrame] = []

    for _, row in selected.iterrows():
        origin_geom = row["origin_geom"]
        print(f"Building driveshed for {row['requested_city']} -> {row['city']}, {row['state']} (pair_id={row['pair_id']})")
        result = dshed.build_h3_driveshed_from_point(
            origin_geom,
            max_travel_minutes=config.max_travel_minutes,
            h3_res=config.h3_res,
            weight_attr=config.weight_attr,
            osm_cache_dir=config.osm_cache_dir,
            osm_force_refresh=config.osm_force_refresh,
            search_buffer_speed_mph=config.search_buffer_speed_mph,
            search_buffer_factor=config.search_buffer_factor,
            search_min_buffer_miles=config.search_min_buffer_miles,
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
            snap_max_k=config.snap_max_k,
        )

        origin_gdf = gpd.GeoDataFrame(
            {
                "pair_id": [int(row["pair_id"])],
                "count": [int(row["count"])],
                "requested_city": [row["requested_city"]],
                "city": [row["city"]],
                "resolved_city": [row["resolved_city"]],
                "state": [row["state"]],
                "category": [row["category"]],
                "origin_h3_cell": [result.origin_h3_cell],
                "origin_h3_cell_graph": [result.origin_h3_cell_graph],
                "h3_res": [result.h3_res],
                "max_travel_minutes": [result.max_travel_minutes],
                "weight_attr": [result.weight_attr],
            },
            geometry=[result.origin_point_wgs84],
            crs="EPSG:4326",
        )
        origin_rows.append(origin_gdf)

        if result.search_polygon_wgs84 is not None:
            search_gdf = gpd.GeoDataFrame(
                {
                    "pair_id": [int(row["pair_id"])],
                    "count": [int(row["count"])],
                    "requested_city": [row["requested_city"]],
                    "city": [row["city"]],
                    "state": [row["state"]],
                    "category": [row["category"]],
                    "h3_res": [result.h3_res],
                    "max_travel_minutes": [result.max_travel_minutes],
                },
                geometry=[result.search_polygon_wgs84],
                crs="EPSG:4326",
            )
            search_rows.append(search_gdf)

        polygon_rows.append(_append_origin_metadata(result.driveshed_gdf, row=row, result=result))
        cell_rows.append(_append_origin_metadata(result.reachable_cells_gdf, row=row, result=result))
        edge_rows.append(_append_origin_metadata(result.reachable_edges_gdf, row=row, result=result))

    layers = [
        ("driveshed_origin_points", gpd.GeoDataFrame(pd.concat(origin_rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")),
        ("driveshed_polygons", gpd.GeoDataFrame(pd.concat(polygon_rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")),
        ("driveshed_cells", gpd.GeoDataFrame(pd.concat(cell_rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")),
        ("driveshed_edges", gpd.GeoDataFrame(pd.concat(edge_rows, ignore_index=True), geometry="geometry", crs="EPSG:4326")),
    ]
    if search_rows:
        layers.append(
            (
                "driveshed_search_polygons",
                gpd.GeoDataFrame(pd.concat(search_rows, ignore_index=True), geometry="geometry", crs="EPSG:4326"),
            )
        )

    driveshed_cells_gdf = next(gdf for layer_name, gdf in layers if layer_name == "driveshed_cells")
    upsampled_cells_gdf = dshed.aggregate_driveshed_cells_to_parent_layers(
        driveshed_cells_gdf,
        target_resolutions=config.upsampled_target_resolutions,
        h3_cell_col="h3_cell",
    )
    if not upsampled_cells_gdf.empty:
        upsampled_polygons_gdf = dshed.dissolve_upsampled_driveshed_cells(upsampled_cells_gdf)
        layers.append(("driveshed_cells_upsampled", upsampled_cells_gdf))
        layers.append(("driveshed_polygons_upsampled", upsampled_polygons_gdf))

    written_layers = hcal.write_layers_to_gpkg(output_gpkg_path, layers=layers)
    return {
        "selected_pairs": selected,
        "written_layers": written_layers,
        "output_gpkg_path": output_gpkg_path,
    }


if __name__ == "__main__":
    vintage = 0
    output_dir = os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing")
    csv_file = os.path.join(os.path.dirname(__file__), "osm_scale_calibration.csv")
    output_gpkg = os.path.join(output_dir, f"h3_driveshed_vintage{vintage}.gpkg")

    cfg = DriveshedTestConfig()
    result = run_driveshed_test(
        csv_path=csv_file,
        output_gpkg_path=output_gpkg,
        config=cfg,
    )
    print("Driveshed test complete.")
    print("GPKG:", result["output_gpkg_path"])
    print("Layers written:", result["written_layers"])
