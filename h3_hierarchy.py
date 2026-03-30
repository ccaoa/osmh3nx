from __future__ import annotations

from typing import Dict, List, Sequence

import geopandas as gpd
import h3
from shapely.geometry import Polygon


def _h3_cell_to_polygon(cell: str) -> Polygon:
    boundary_latlng = h3.cell_to_boundary(cell)
    boundary_lnglat = [(lng, lat) for (lat, lng) in boundary_latlng]
    return Polygon(boundary_lnglat)


def aggregate_cells_to_parent(
    cells: Sequence[str],
    target_res: int,
) -> List[str]:
    """
    Roll a set of H3 cells up to unique parent cells at a coarser resolution.

    Parameters
    ----------
    cells:
        Input H3 cells, all assumed to be valid H3 indexes.
    target_res:
        Coarser target resolution. Must be <= the resolution of every input cell.
    """
    parents: set[str] = set()
    for cell in cells:
        cell_res = int(h3.get_resolution(cell))
        if target_res > cell_res:
            raise ValueError(
                f"target_res={target_res} is finer than input cell resolution {cell_res} for cell {cell}."
            )
        parents.add(h3.cell_to_parent(cell, target_res))
    return sorted(parents)


def aggregate_cells_to_parent_records(
    cells: Sequence[str],
    target_res: int,
) -> List[Dict[str, object]]:
    """
    Roll input cells to unique parent cells and return one record per parent.

    Each record includes:
      - `h3_cell`: the parent cell id
      - `h3_res`: the parent resolution
      - `n_child_cells`: number of input cells collapsed into that parent
      - `geometry`: parent cell polygon in EPSG:4326
    """
    parent_to_count: Dict[str, int] = {}
    for cell in cells:
        cell_res = int(h3.get_resolution(cell))
        if target_res > cell_res:
            raise ValueError(
                f"target_res={target_res} is finer than input cell resolution {cell_res} for cell {cell}."
            )
        parent = h3.cell_to_parent(cell, target_res)
        parent_to_count[parent] = parent_to_count.get(parent, 0) + 1

    records: List[Dict[str, object]] = []
    for parent in sorted(parent_to_count):
        records.append(
            {
                "h3_cell": parent,
                "h3_res": int(target_res),
                "n_child_cells": int(parent_to_count[parent]),
                "geometry": _h3_cell_to_polygon(parent),
            }
        )
    return records


def aggregate_cells_to_parent_gdf(
    cells: Sequence[str],
    target_res: int,
) -> gpd.GeoDataFrame:
    """
    Roll input cells to unique parent cells and return them as polygons in EPSG:4326.
    """
    records = aggregate_cells_to_parent_records(cells, target_res)
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")


def aggregate_h3_gdf_to_parent_gdf(
    h3_gdf: gpd.GeoDataFrame,
    target_res: int,
    *,
    h3_cell_col: str = "h3_cell",
) -> gpd.GeoDataFrame:
    """
    Roll H3 cells from an existing GeoDataFrame up to unique parent polygons.

    Parameters
    ----------
    h3_gdf:
        Input GeoDataFrame containing H3 cell ids.
    target_res:
        Coarser target resolution. Must be <= the resolution of every input cell.
    h3_cell_col:
        Column containing H3 cell ids. Defaults to `h3_cell`, which is the
        standard field name used by this project.
    """
    if h3_cell_col not in h3_gdf.columns:
        raise ValueError(f"Column '{h3_cell_col}' not found in input GeoDataFrame.")

    cells = [str(cell) for cell in h3_gdf[h3_cell_col].dropna().tolist()]
    return aggregate_cells_to_parent_gdf(cells, target_res)
