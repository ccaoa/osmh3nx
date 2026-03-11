from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import wkt
from shapely.geometry import Point

# Ensure imports work when this script lives under ./offline.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import network_h3 as hnetx
import network_osm as onetx


@dataclass(frozen=True)
class OfflinePrepConfig:
    h3_resolutions: Tuple[int, ...] = (7, 8, 9, 10)
    bbox_buffer_miles: float = 15.0
    sample_miles: float = 0.25
    combine_parallel: str = "mean"
    enforce_min_step_time: bool = True
    v_max_mph: float = 35.0
    floor_speed_source: str = "osm_median"
    min_osm_speed_mph: float = 10.0 / hnetx.KM_PER_MILE
    preserve_way_geometry: bool = True
    way_cell_refine_max_depth: int = 18
    max_pairs: int | None = None
    overwrite: bool = False


def _required_columns() -> List[str]:
    return ["count", "city", "state", "category", "origin", "destination"]


def _load_od_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in _required_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    out = df.copy()
    out["pair_id"] = out["count"].astype(int)
    out["origin_geom"] = out["origin"].apply(wkt.loads)
    out["destination_geom"] = out["destination"].apply(wkt.loads)

    bad_origin = out["origin_geom"].apply(lambda g: not isinstance(g, Point))
    bad_dest = out["destination_geom"].apply(lambda g: not isinstance(g, Point))
    if bool(bad_origin.any()) or bool(bad_dest.any()):
        raise ValueError("Origin and destination must be WKT POINT geometries.")

    return out.sort_values("pair_id").reset_index(drop=True)


def _save_pickle_graph(G: nx.Graph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)


def _prep_pair(
    row: pd.Series,
    out_dir: Path,
    config: OfflinePrepConfig,
) -> Dict[str, Any]:
    pair_id = int(row["pair_id"])
    city = str(row["city"])
    state = str(row["state"])
    category = str(row["category"])
    origin = row["origin_geom"]
    destination = row["destination_geom"]

    pair_dir = out_dir / f"pair_{pair_id:05d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    square_poly, buffer_poly = onetx.build_pair_square_and_buffer(
        origin,
        destination,
        buffer_miles=config.bbox_buffer_miles,
    )

    graphml_path = pair_dir / "osm_drive.graphml"
    if graphml_path.exists() and not config.overwrite:
        G_osm = ox.load_graphml(str(graphml_path))
        osm_source = "graphml"
    else:
        G_osm = onetx.download_osm_drive_graph_for_polygon(buffer_poly)
        ox.save_graphml(G_osm, filepath=str(graphml_path))
        osm_source = "download"

    h3_outputs: List[Dict[str, Any]] = []
    for h3_res in config.h3_resolutions:
        h3_graph_path = pair_dir / f"h3_drive_res{int(h3_res)}.pkl"
        if h3_graph_path.exists() and not config.overwrite:
            with h3_graph_path.open("rb") as f:
                H_h3 = pickle.load(f)
            h3_source = "pickle"
        else:
            H_h3 = hnetx.build_h3_travel_graph_from_osm(
                G_osm,
                h3_res=int(h3_res),
                weight_attr="travel_time",
                sample_miles=config.sample_miles,
                combine_parallel=config.combine_parallel,
                enforce_min_step_time=config.enforce_min_step_time,
                v_max_mph=config.v_max_mph,
                floor_speed_source=config.floor_speed_source,
                min_osm_speed_mph=config.min_osm_speed_mph,
                preserve_way_geometry=config.preserve_way_geometry,
                way_cell_refine_max_depth=config.way_cell_refine_max_depth,
            )
            _save_pickle_graph(H_h3, h3_graph_path)
            h3_source = "build"

        h3_outputs.append(
            {
                "h3_res": int(h3_res),
                "h3_graph_path": str(h3_graph_path),
                "h3_graph_nodes": int(H_h3.number_of_nodes()),
                "h3_graph_edges": int(H_h3.number_of_edges()),
                "h3_source": h3_source,
            }
        )

    meta = {
        "pair_id": pair_id,
        "city": city,
        "state": state,
        "category": category,
        "origin_wkt": str(row["origin"]),
        "destination_wkt": str(row["destination"]),
        "square_wkt": square_poly.wkt,
        "buffer_wkt": buffer_poly.wkt,
        "osm_graphml_path": str(graphml_path),
        "osm_nodes": int(G_osm.number_of_nodes()),
        "osm_edges": int(G_osm.number_of_edges()),
        "osm_source": osm_source,
        "h3_graphs": h3_outputs,
    }
    with (pair_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def prepare_offline_bundle(
    *,
    csv_path: str,
    out_dir: str,
    config: OfflinePrepConfig,
) -> Dict[str, Any]:
    ox.settings.use_cache = True
    ox.settings.log_console = True

    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    od_pairs = _load_od_pairs(csv_path)
    if config.max_pairs is not None:
        od_pairs = od_pairs.head(int(config.max_pairs)).copy()
    manifests: List[Dict[str, Any]] = []
    for i, row in od_pairs.iterrows():
        pair_id = int(row["pair_id"])
        print(f"[{i + 1}/{len(od_pairs)}] preparing pair_id={pair_id}")
        manifests.append(_prep_pair(row, out_path, config))

    run_manifest = {
        "csv_path": str(Path(csv_path).resolve()),
        "out_dir": str(out_path),
        "config": asdict(config),
        "pair_count": int(len(manifests)),
        "pairs": manifests,
    }
    manifest_path = out_path / "offline_bundle_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(run_manifest, f, indent=2)
    return {"manifest_path": str(manifest_path), "pair_count": int(len(manifests))}


def _parse_resolutions(txt: str) -> Tuple[int, ...]:
    vals = [int(x.strip()) for x in txt.split(",") if x.strip()]
    if not vals:
        raise ValueError("At least one H3 resolution is required.")
    return tuple(vals)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare offline OSM+H3 routing bundle.")
    p.add_argument("--csv", required=True, help="Path to OD calibration CSV.")
    p.add_argument("--out-dir", required=True, help="Output directory for offline bundle.")
    p.add_argument("--resolutions", default="7,8,9,10", help="Comma-separated H3 resolutions.")
    p.add_argument("--buffer-miles", type=float, default=15.0)
    p.add_argument("--sample-miles", type=float, default=0.25)
    p.add_argument("--combine-parallel", default="mean", choices=["min", "mean"])
    p.add_argument("--v-max-mph", type=float, default=35.0)
    p.add_argument("--floor-speed-source", default="osm_median", choices=["vmax", "osm_median"])
    p.add_argument("--min-osm-speed-mph", type=float, default=10.0 / hnetx.KM_PER_MILE)
    p.add_argument("--no-step-floor", action="store_true")
    p.add_argument("--no-preserve-way-geometry", action="store_true")
    p.add_argument("--way-refine-max-depth", type=int, default=18)
    p.add_argument("--max-pairs", type=int, default=None, help="Optional cap for number of OD pairs to prepare.")
    p.add_argument("--overwrite", action="store_true")
    return p


if __name__ == "__main__":
    parser = _build_arg_parser()
    args = parser.parse_args()

    cfg = OfflinePrepConfig(
        h3_resolutions=_parse_resolutions(args.resolutions),
        bbox_buffer_miles=float(args.buffer_miles),
        sample_miles=float(args.sample_miles),
        combine_parallel=str(args.combine_parallel),
        enforce_min_step_time=not bool(args.no_step_floor),
        v_max_mph=float(args.v_max_mph),
        floor_speed_source=str(args.floor_speed_source),
        min_osm_speed_mph=float(args.min_osm_speed_mph),
        preserve_way_geometry=not bool(args.no_preserve_way_geometry),
        way_cell_refine_max_depth=int(args.way_refine_max_depth),
        max_pairs=int(args.max_pairs) if args.max_pairs is not None else None,
        overwrite=bool(args.overwrite),
    )

    result = prepare_offline_bundle(
        csv_path=str(args.csv),
        out_dir=str(args.out_dir),
        config=cfg,
    )
    print("Offline bundle complete.")
    print("Manifest:", result["manifest_path"])
    print("Pairs:", result["pair_count"])
