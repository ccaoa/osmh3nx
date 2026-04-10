from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Literal, Sequence

import geopandas as gpd
import osmnx as ox
import pandas as pd

try:
    from _bootstrap import default_routing_output_dir, ensure_src_on_path, repo_root
except ImportError:
    from scripts._bootstrap import default_routing_output_dir, ensure_src_on_path, repo_root

ensure_src_on_path()

from osmh3nx import calibrate as calx
from osmh3nx import driveshed as dshed
from osmh3nx.batch import run_batch_drivesheds
from osmh3nx.data import load_od_pairs
from osmh3nx.io import write_layers_to_gpkg, write_table_sidecar

REPO_CACHE_DIR: str = str(repo_root() / "cache")


@dataclass(frozen=True)
class AllOriginsDriveshedConfig:
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME
    calibration_profile_overrides: Dict[str, Any] | None = None
    h3_res: int = 10
    upsampled_target_resolutions: Sequence[int] = (9, 8)
    max_travel_minutes: float = 20.0
    weight_attr: str = dshed.DEFAULT_H3_WEIGHT_ATTR
    search_buffer_speed_mph: float = 60.0
    search_buffer_factor: float = dshed.DEFAULT_SEARCH_BUFFER_FACTOR
    search_min_buffer_miles: float = 2.0
    graph_group_cols: Sequence[str] = ()
    share_graph_for_all_origins: bool = False
    shared_graph_buffer_miles: float = dshed.DEFAULT_SHARED_GRAPH_BUFFER_MILES
    lookup_table_format: Literal["csv", "parquet"] = "csv"
    snap_max_k: int = 10
    osm_cache_dir: str | None = REPO_CACHE_DIR
    osm_force_refresh: bool = False


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now_text()}] {message}", flush=True)


def _build_origin_inputs(od_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    origins = od_pairs.copy()
    origins["requested_city"] = origins["city"]
    origins["resolved_city"] = origins["city"]
    origins["origin_wkt"] = origins["origin"]
    origins["destination_wkt"] = origins["destination"]

    return gpd.GeoDataFrame(
        origins[
            [
                "pair_id",
                "count",
                "city",
                "state",
                "category",
                "requested_city",
                "resolved_city",
                "origin_wkt",
                "destination_wkt",
            ]
        ].copy(),
        geometry=origins["origin_geom"],
        crs="EPSG:4326",
    )


def run_all_origins_driveshed_test(
    *,
    csv_path: str,
    output_gpkg_path: str,
    config: AllOriginsDriveshedConfig,
) -> Dict[str, Any]:
    run_started = perf_counter()
    _log(f"Loading OD pairs from {csv_path}")
    od_pairs = load_od_pairs(csv_path)
    origin_inputs = _build_origin_inputs(od_pairs)
    _log(f"Loaded {len(origin_inputs)} origins from calibration CSV")
    _log(
        "Run configuration "
        f"(profile={config.calibration_profile_name}, weight_attr={config.weight_attr}, "
        f"osm_cache_dir={config.osm_cache_dir}, osm_force_refresh={config.osm_force_refresh}, "
        f"snap_max_k={config.snap_max_k}, graph_group_cols={tuple(config.graph_group_cols)}, "
        f"share_graph_for_all_origins={config.share_graph_for_all_origins}, "
        f"shared_graph_buffer_miles={config.shared_graph_buffer_miles}, "
        f"lookup_table_format={config.lookup_table_format})"
    )
    _log(
        "Starting batch driveshed build "
        f"(max_travel_minutes={config.max_travel_minutes}, h3_res={config.h3_res}, "
        f"upsampled_target_resolutions={tuple(config.upsampled_target_resolutions)})"
    )

    result = run_batch_drivesheds(
        origin_inputs,
        origin_id_col="pair_id",
        graph_group_cols=config.graph_group_cols,
        share_graph_for_all_origins=config.share_graph_for_all_origins,
        geometry_col="geometry",
        max_travel_minutes=config.max_travel_minutes,
        h3_res=config.h3_res,
        weight_attr=config.weight_attr,
        calibration_profile_name=config.calibration_profile_name,
        calibration_profile_overrides=config.calibration_profile_overrides,
        osm_cache_dir=config.osm_cache_dir,
        osm_force_refresh=config.osm_force_refresh,
        search_buffer_speed_mph=config.search_buffer_speed_mph,
        search_buffer_factor=config.search_buffer_factor,
        search_min_buffer_miles=config.search_min_buffer_miles,
        shared_graph_buffer_miles=config.shared_graph_buffer_miles,
        snap_max_k=config.snap_max_k,
        upsampled_target_resolutions=config.upsampled_target_resolutions,
    )
    _log(
        "Batch driveshed build finished "
        f"(origins={len(result.origins_gdf)}, unique_cells={len(result.driveshed_cells_unique_gdf)}, "
        f"lookup_rows={len(result.driveshed_cell_lookup_df)})"
    )
    status_counts = (
        result.origins_gdf["status"].fillna("missing_status").value_counts().to_dict()
        if "status" in result.origins_gdf.columns
        else {}
    )
    _log(f"Origin status counts: {status_counts}")

    layers = [
        ("driveshed_origin_points", result.origins_gdf),
        ("driveshed_search_polygons", result.search_polygons_gdf),
        ("driveshed_cells_unique", result.driveshed_cells_unique_gdf),
    ]
    _log(f"Writing GeoPackage to {output_gpkg_path}")
    written_layers = write_layers_to_gpkg(output_gpkg_path, layers=layers)
    lookup_path = os.path.splitext(output_gpkg_path)[0] + f"_cell_lookup.{config.lookup_table_format}"
    _log(f"Writing lookup sidecar table to {lookup_path}")
    write_table_sidecar(
        result.driveshed_cell_lookup_df,
        lookup_path,
        table_format=config.lookup_table_format,
    )
    elapsed_total = perf_counter() - run_started
    _log(
        f"GeoPackage write finished with {len(written_layers)} non-empty layers "
        f"(elapsed_total_sec={elapsed_total:.1f})"
    )

    return {
        "n_origins": int(len(origin_inputs)),
        "output_gpkg_path": output_gpkg_path,
        "lookup_table_path": lookup_path,
        "written_layers": written_layers,
    }


if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.cache_folder = REPO_CACHE_DIR

    vintage = 0
    output_dir = str(default_routing_output_dir())
    csv_file = str(repo_root() / "osm_scale_calibration.csv")
    output_gpkg = os.path.join(
        output_dir, f"h3_driveshed_all_origins_vintage{vintage}.gpkg"
    )

    cfg = AllOriginsDriveshedConfig()
    _log("All-origins driveshed test starting")
    _log(f"Repo cache directory: {REPO_CACHE_DIR}")
    _log(f"OSMnx cache folder: {ox.settings.cache_folder}")
    _log(f"Output GeoPackage: {output_gpkg}")
    run_result = run_all_origins_driveshed_test(
        csv_path=csv_file,
        output_gpkg_path=output_gpkg,
        config=cfg,
    )
    _log("All-origins driveshed test complete")
    _log(f"Origins processed: {run_result['n_origins']}")
    _log(f"GPKG: {run_result['output_gpkg_path']}")
    _log(f"Lookup table: {run_result['lookup_table_path']}")
    _log(f"Layers written: {run_result['written_layers']}")
