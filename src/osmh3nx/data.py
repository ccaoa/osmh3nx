from __future__ import annotations

from typing import Any, Dict, List

import geopandas as gpd
import pandas as pd
from shapely import wkt
from shapely.geometry import Point


def required_od_columns() -> List[str]:
    return ["count", "city", "state", "category", "origin", "destination"]


def load_od_pairs(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in required_od_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in calibration CSV: {missing}")

    out = df.copy()
    out["pair_id"] = out["count"].astype(int)
    out["origin_geom"] = out["origin"].apply(wkt.loads)
    out["destination_geom"] = out["destination"].apply(wkt.loads)

    bad_origin = out["origin_geom"].apply(lambda g: not isinstance(g, Point))
    bad_dest = out["destination_geom"].apply(lambda g: not isinstance(g, Point))
    if bool(bad_origin.any()) or bool(bad_dest.any()):
        raise ValueError("Origin and destination WKT must parse to POINT geometries.")

    out = out.sort_values("pair_id").reset_index(drop=True)
    return out


def build_od_points_gdf(od_pairs: pd.DataFrame) -> gpd.GeoDataFrame:
    rows: List[Dict[str, Any]] = []
    for _, r in od_pairs.iterrows():
        base = {
            "pair_id": int(r["pair_id"]),
            "count": int(r["count"]),
            "city": r["city"],
            "state": r["state"],
            "category": r["category"],
            "origin": r["origin"],
            "destination": r["destination"],
        }
        rows.append(
            {
                **base,
                "point_role": "origin",
                "point_wkt": r["origin"],
                "geometry": r["origin_geom"],
            }
        )
        rows.append(
            {
                **base,
                "point_role": "destination",
                "point_wkt": r["destination"],
                "geometry": r["destination_geom"],
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")
