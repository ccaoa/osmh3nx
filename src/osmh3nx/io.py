from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import geopandas as gpd


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
