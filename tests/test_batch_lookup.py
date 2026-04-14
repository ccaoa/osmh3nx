from __future__ import annotations

import geopandas as gpd
import pandas as pd

from osmh3nx import batch as batchx


def test_select_driveshed_cells_from_lookup_filters_by_origin(
    sample_unique_cells_gdf: gpd.GeoDataFrame,
    sample_lookup_df: pd.DataFrame,
) -> None:
    selected = batchx.select_driveshed_cells_from_lookup(
        sample_unique_cells_gdf,
        sample_lookup_df,
        origin_ids=["origin_a"],
    )

    assert set(selected["h3_res"]) == {9, 10}
    assert len(selected) == 3


def test_select_origins_for_driveshed_cell_returns_matching_rows(
    sample_h3_cells: dict[str, str],
    sample_lookup_df: pd.DataFrame,
) -> None:
    selected = batchx.select_origins_for_driveshed_cell(
        sample_lookup_df,
        h3_cell=sample_h3_cells["neighbor_cell"],
        h3_res=10,
    )

    assert set(selected["origin_id"]) == {"origin_a", "origin_b"}
    assert (selected["h3_res"] == 10).all()


def test_dissolve_driveshed_cells_from_lookup_groups_by_origin(
    sample_unique_cells_gdf: gpd.GeoDataFrame,
    sample_lookup_df: pd.DataFrame,
) -> None:
    dissolved = batchx.dissolve_driveshed_cells_from_lookup(
        sample_unique_cells_gdf,
        sample_lookup_df,
        dissolve_cols=["origin_id"],
    )

    assert set(dissolved["origin_id"]) == {"origin_a", "origin_b"}
    assert dissolved["n_cells"].min() >= 1
    assert dissolved.geometry.notna().all()


def test_dissolve_driveshed_cells_from_lookup_respects_query(
    sample_unique_cells_gdf: gpd.GeoDataFrame,
    sample_lookup_df: pd.DataFrame,
) -> None:
    dissolved = batchx.dissolve_driveshed_cells_from_lookup(
        sample_unique_cells_gdf,
        sample_lookup_df,
        query="h3_res == 10",
    )

    assert len(dissolved) == 1
    assert int(dissolved["h3_res"].iloc[0]) == 10


def test_select_origins_for_driveshed_cell_empty_input_returns_empty_df() -> None:
    selected = batchx.select_origins_for_driveshed_cell(
        pd.DataFrame(columns=["origin_id", "h3_cell", "h3_res"]),
        h3_cell="8a0000000000000",
    )

    assert selected.empty
