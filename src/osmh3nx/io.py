from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Sequence, Tuple

import geopandas as gpd
import pandas as pd

TableOutputFormat = Literal["csv", "parquet"]


def write_layers_to_gpkg(
    gpkg_path: str,
    *,
    layers: Sequence[Tuple[str, gpd.GeoDataFrame]],
) -> List[str]:
    parent = os.path.dirname(gpkg_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if os.path.exists(gpkg_path):
        os.remove(gpkg_path)

    written: List[str] = []
    mode = "w"
    for layer_name, gdf in layers:
        if gdf.empty:
            continue
        out_gdf = gdf
        if out_gdf.crs is None:
            out_gdf = out_gdf.set_crs("EPSG:4326")
        out_gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG", mode=mode)
        written.append(layer_name)
        mode = "a"
    return written


def write_table_sidecar(
    table: pd.DataFrame,
    output_path: str | Path,
    *,
    table_format: TableOutputFormat = "parquet",
) -> Path:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if table_format == "csv":
        table.to_csv(out_path, index=False)
        return out_path

    if table_format == "parquet":
        try:
            table.to_parquet(out_path, index=False)
        except ImportError as exc:
            raise ImportError(
                "Parquet output requires an optional parquet engine such as 'pyarrow'."
            ) from exc
        return out_path

    raise ValueError(
        f"Unsupported table_format '{table_format}'. Expected 'csv' or 'parquet'."
    )
