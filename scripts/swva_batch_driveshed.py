from __future__ import annotations

import json
import os
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, Literal, Sequence

import geopandas as gpd
import osmnx as ox
import pandas as pd

try:
    from _bootstrap import ensure_src_on_path, repo_root
except ImportError:
    from scripts._bootstrap import ensure_src_on_path, repo_root

ensure_src_on_path()

from osmh3nx import calibrate as calx
from osmh3nx import driveshed as dshed
from osmh3nx.batch import run_batch_drivesheds
from osmh3nx.io import write_layers_to_gpkg, write_table_sidecar

REPO_CACHE_DIR: str = str(repo_root() / "cache")
SCRIPT_CSV_PATH: str = str(repo_root() / "scripts" / "swva_ece_wls_20260315.csv")
DEFAULT_OSM_NETWORK_EDGES_BASENAME: str = "swva_osm_drive_network_edges.geojson"
OSM_NETWORK_EXPORT_META_SUFFIX: str = ".meta.json"


@dataclass(frozen=True)
class SWVABatchDriveshedConfig:
    calibration_profile_name: str = calx.DEFAULT_PROFILE_NAME
    calibration_profile_overrides: Dict[str, Any] | None = None
    h3_res: int = 10
    upsampled_target_resolutions: Sequence[int] = (9, 8)
    max_travel_minutes: float = 20.0
    weight_attr: str = dshed.DEFAULT_H3_WEIGHT_ATTR
    share_graph_for_all_origins: bool = True
    shared_graph_buffer_miles: float = dshed.DEFAULT_SHARED_GRAPH_BUFFER_MILES
    lookup_table_format: Literal["csv", "parquet"] = "csv"
    search_buffer_speed_mph: float = 60.0
    search_buffer_factor: float = dshed.DEFAULT_SEARCH_BUFFER_FACTOR
    search_min_buffer_miles: float = 2.0
    snap_max_k: int = 10
    osm_cache_dir: str | None = REPO_CACHE_DIR
    osm_force_refresh: bool = False


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now_text()}] {message}", flush=True)


def load_swva_inputs(csv_path: str) -> gpd.GeoDataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"id", "latitude", "longitude"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise ValueError(f"SWVA CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["id"] = out["id"].astype(str)
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    return gpd.GeoDataFrame(
        out[["id", "latitude", "longitude"]].copy(),
        geometry=gpd.points_from_xy(out["longitude"], out["latitude"]),
        crs="EPSG:4326",
    )


def run_swva_batch_driveshed_test(
    *,
    csv_path: str,
    output_gpkg_path: str,
    config: SWVABatchDriveshedConfig,
) -> Dict[str, Any]:
    run_started = perf_counter()
    _log(f"Loading SWVA driveshed inputs from {csv_path}")
    origins_gdf = load_swva_inputs(csv_path)
    _log(f"Loaded {len(origins_gdf)} SWVA origin points")
    _log(
        "Run configuration "
        f"(profile={config.calibration_profile_name}, weight_attr={config.weight_attr}, "
        f"share_graph_for_all_origins={config.share_graph_for_all_origins}, "
        f"shared_graph_buffer_miles={config.shared_graph_buffer_miles}, "
        f"lookup_table_format={config.lookup_table_format}, "
        f"osm_cache_dir={config.osm_cache_dir}, osm_force_refresh={config.osm_force_refresh}, "
        f"snap_max_k={config.snap_max_k})"
    )
    _log(
        "Starting SWVA batch driveshed build "
        f"(max_travel_minutes={config.max_travel_minutes}, h3_res={config.h3_res}, "
        f"upsampled_target_resolutions={tuple(config.upsampled_target_resolutions)})"
    )

    result = run_batch_drivesheds(
        origins_gdf,
        origin_id_col="id",
        geometry_col="geometry",
        max_travel_minutes=config.max_travel_minutes,
        h3_res=config.h3_res,
        weight_attr=config.weight_attr,
        calibration_profile_name=config.calibration_profile_name,
        calibration_profile_overrides=config.calibration_profile_overrides,
        share_graph_for_all_origins=config.share_graph_for_all_origins,
        shared_graph_buffer_miles=config.shared_graph_buffer_miles,
        osm_cache_dir=config.osm_cache_dir,
        osm_force_refresh=config.osm_force_refresh,
        search_buffer_speed_mph=config.search_buffer_speed_mph,
        search_buffer_factor=config.search_buffer_factor,
        search_min_buffer_miles=config.search_min_buffer_miles,
        snap_max_k=config.snap_max_k,
        upsampled_target_resolutions=config.upsampled_target_resolutions,
    )
    _log(
        "SWVA batch driveshed build finished "
        f"(origins={len(result.origins_gdf)}, unique_cells={len(result.driveshed_cells_unique_gdf)}, "
        f"lookup_rows={len(result.driveshed_cell_lookup_df)}, "
        f"shared_graphs={len(result.graph_contexts)})"
    )
    status_counts = (
        result.origins_gdf["status"].fillna("missing_status").value_counts().to_dict()
        if "status" in result.origins_gdf.columns
        else {}
    )
    _log(f"Origin status counts: {status_counts}")
    output_dir = Path(output_gpkg_path).expanduser().resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_osm_network_exports_if_missing(
        graph_contexts=result.graph_contexts,
        output_dir=output_dir,
    )

    layers = [
        ("swva_origin_points", result.origins_gdf),
        ("swva_search_polygons", result.search_polygons_gdf),
        ("swva_driveshed_cells_unique", result.driveshed_cells_unique_gdf),
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
        "n_origins": int(len(origins_gdf)),
        "output_gpkg_path": output_gpkg_path,
        "lookup_table_path": lookup_path,
        "written_layers": written_layers,
    }


def _write_osm_network_exports_if_missing(
    *,
    graph_contexts: Dict[str, Any],
    output_dir: Path,
) -> None:
    if not graph_contexts:
        _log("No graph contexts available for OSM network export")
        return

    if len(graph_contexts) == 1:
        graph_group_id, context = next(iter(graph_contexts.items()))
        output_path = output_dir / DEFAULT_OSM_NETWORK_EDGES_BASENAME
        _write_osm_edges_geojson_if_missing(
            graph_group_id=graph_group_id,
            osm_graph=context.osm_graph,
            search_polygon_wgs84=context.search_polygon_wgs84,
            output_path=output_path,
        )
        return

    for graph_group_id, context in graph_contexts.items():
        output_path = output_dir / f"swva_osm_drive_network_{graph_group_id}_edges.geojson"
        _write_osm_edges_geojson_if_missing(
            graph_group_id=graph_group_id,
            osm_graph=context.osm_graph,
            search_polygon_wgs84=context.search_polygon_wgs84,
            output_path=output_path,
        )


def _write_osm_edges_geojson_if_missing(
    *,
    graph_group_id: str,
    osm_graph: Any,
    search_polygon_wgs84: Any,
    output_path: Path,
) -> None:
    if osm_graph is None or osm_graph.number_of_edges() == 0:
        _log(f"No OSM graph available for edge export (graph_group_id={graph_group_id})")
        return

    metadata = _build_osm_export_metadata(
        graph_group_id=graph_group_id,
        osm_graph=osm_graph,
        search_polygon_wgs84=search_polygon_wgs84,
    )
    metadata_path = _osm_export_metadata_path(output_path)
    if output_path.exists() and metadata_path.exists():
        existing_metadata = _read_osm_export_metadata(metadata_path)
        if existing_metadata == metadata:
            _log(f"OSM edge network already exists, skipping export: {output_path}")
            return

    if output_path.exists():
        _log(
            "Existing OSM edge export does not match the current shared graph, "
            f"rewriting: {output_path}"
        )

    _log(f"Exporting OSM edge network GeoJSON for graph_group_id={graph_group_id} to {output_path}")
    edges_gdf = ox.graph_to_gdfs(
        osm_graph,
        nodes=False,
        edges=True,
        fill_edge_geometry=True,
    ).reset_index()
    edges_gdf = gpd.GeoDataFrame(edges_gdf, geometry="geometry", crs=edges_gdf.crs).to_crs(
        "EPSG:4326"
    )
    edges_gdf.to_file(output_path, driver="GeoJSON")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _osm_export_metadata_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + OSM_NETWORK_EXPORT_META_SUFFIX)


def _read_osm_export_metadata(metadata_path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _build_osm_export_metadata(
    *,
    graph_group_id: str,
    osm_graph: Any,
    search_polygon_wgs84: Any,
) -> Dict[str, Any]:
    polygon_bounds = None
    if search_polygon_wgs84 is not None:
        polygon_bounds = [float(value) for value in search_polygon_wgs84.bounds]

    return {
        "graph_group_id": str(graph_group_id),
        "osm_nodes": int(osm_graph.number_of_nodes()),
        "osm_edges": int(osm_graph.number_of_edges()),
        "search_polygon_bounds_wgs84": polygon_bounds,
    }


if __name__ == "__main__":
    ox.settings.use_cache = True
    ox.settings.cache_folder = REPO_CACHE_DIR

    vintage = 1
    output_dir = os.path.expanduser(r"~/OneDrive - NACCRRA\Documents\skratch\routing")
    output_gpkg = os.path.join(output_dir, f"swva_batch_driveshed_vintage{vintage}.gpkg")

    cfg = SWVABatchDriveshedConfig()
    _log("SWVA batch driveshed test starting")
    _log(f"Repo cache directory: {REPO_CACHE_DIR}")
    _log(f"OSMnx cache folder: {ox.settings.cache_folder}")
    _log(f"Input CSV: {SCRIPT_CSV_PATH}")
    _log(f"Output GeoPackage: {output_gpkg}")
    run_result = run_swva_batch_driveshed_test(
        csv_path=SCRIPT_CSV_PATH,
        output_gpkg_path=output_gpkg,
        config=cfg,
    )
    _log("SWVA batch driveshed test complete")
    _log(f"Origins processed: {run_result['n_origins']}")
    _log(f"GPKG: {run_result['output_gpkg_path']}")
    _log(f"Lookup table: {run_result['lookup_table_path']}")
    _log(f"Layers written: {run_result['written_layers']}")
