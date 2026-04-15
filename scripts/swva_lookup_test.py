from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

try:
    from _bootstrap import default_routing_output_dir, ensure_src_on_path
except ImportError:
    from scripts._bootstrap import default_routing_output_dir, ensure_src_on_path

ensure_src_on_path()

from osmh3nx import select_origins_for_driveshed_cell

TARGET_H3_CELL: str = "8a2a8a905aeffff"
VINTAGE: int = 2
OUTPUT_DIR: str = str(default_routing_output_dir())
GPKG_PATH: str = os.path.join(OUTPUT_DIR, f"swva_batch_driveshed_vintage{VINTAGE}.gpkg")
LOOKUP_CSV_PATH: str = os.path.join(
    OUTPUT_DIR,
    f"swva_batch_driveshed_vintage{VINTAGE}_cell_lookup.csv",
)
UNIQUE_CELLS_LAYER: str = "swva_driveshed_cells_unique"


def _utc_now_text() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _log(message: str) -> None:
    print(f"[{_utc_now_text()}] {message}", flush=True)


def _print_table_preview(table: pd.DataFrame, *, n: int = 10) -> None:
    if table.empty:
        print("(no matching rows)", flush=True)
        return
    print(table.head(n).to_string(index=False), flush=True)


def _build_qgis_wlsid_query(origin_ids: pd.Series) -> str:
    unique_ids = [
        str(value) for value in origin_ids.dropna().astype(str).drop_duplicates()
    ]
    quoted_ids = ", ".join(f"'{value}'" for value in unique_ids)
    return f'"id" IN ({quoted_ids})'


def main() -> None:
    gpkg_path = Path(GPKG_PATH).expanduser().resolve()
    lookup_csv_path = Path(LOOKUP_CSV_PATH).expanduser().resolve()

    _log(f"Loading unique H3 cells from {gpkg_path} layer '{UNIQUE_CELLS_LAYER}'")
    unique_cells_gdf = gpd.read_file(gpkg_path, layer=UNIQUE_CELLS_LAYER)
    _log(f"Loaded {len(unique_cells_gdf)} unique H3 cell rows")

    _log(f"Loading lookup CSV from {lookup_csv_path}")
    cell_lookup_df = pd.read_csv(lookup_csv_path)
    _log(f"Loaded {len(cell_lookup_df)} lookup rows")

    _log(f"Selecting all origins whose driveshed crosses H3 cell {TARGET_H3_CELL}")
    matching_rows = select_origins_for_driveshed_cell(
        cell_lookup_df,
        h3_cell=TARGET_H3_CELL,
    )
    _log(f"Lookup returned {len(matching_rows)} matching rows")

    matching_unique_cells = unique_cells_gdf.loc[
        unique_cells_gdf["h3_cell"].astype(str) == TARGET_H3_CELL
    ].copy()
    _log(
        f"Unique-cell GeoDataFrame contains {len(matching_unique_cells)} row(s) "
        f"for target H3 cell {TARGET_H3_CELL}"
    )

    _log("Previewing the first 10 matching lookup rows")
    _print_table_preview(matching_rows, n=10)

    qgis_query = _build_qgis_wlsid_query(matching_rows["origin_id"])
    _log("QGIS selection query for matching SWVA points")
    print(qgis_query, flush=True)


if __name__ == "__main__":
    main()
