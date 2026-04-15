from __future__ import annotations

import geopandas as gpd
import pytest

from osmh3nx import batch as batchx
from osmh3nx import spatial as spatx


def test_create_convex_hull_returns_single_polygon(
    sample_points_gdf: gpd.GeoDataFrame,
) -> None:
    hull_gdf = spatx.create_convex_hull(sample_points_gdf)

    assert len(hull_gdf) == 1
    assert hull_gdf.crs == sample_points_gdf.crs
    assert hull_gdf.geometry.iloc[0].geom_type == "Polygon"
    assert hull_gdf.geometry.iloc[0].area > 0


def test_create_buffered_convex_hull_grows_area(
    sample_points_gdf: gpd.GeoDataFrame,
) -> None:
    hull_gdf = spatx.create_convex_hull(sample_points_gdf)
    buffered_gdf = spatx.create_buffered_convex_hull(
        sample_points_gdf, buffer_miles=1.0
    )

    proj_crs = hull_gdf.estimate_utm_crs() or "EPSG:3857"
    hull_area = hull_gdf.to_crs(proj_crs).geometry.iloc[0].area
    buffered_area = buffered_gdf.to_crs(proj_crs).geometry.iloc[0].area

    assert buffered_area > hull_area


def test_create_buffered_convex_hull_rejects_negative_buffer(
    sample_points_gdf: gpd.GeoDataFrame,
) -> None:
    with pytest.raises(ValueError, match="buffer_miles must be >= 0"):
        spatx.create_buffered_convex_hull(sample_points_gdf, buffer_miles=-1.0)


def test_build_points_convex_hull_search_polygon_returns_polygon(
    sample_points_gdf: gpd.GeoDataFrame,
) -> None:
    polygon = batchx.build_points_convex_hull_search_polygon(
        sample_points_gdf,
        buffer_miles=0.5,
    )

    assert polygon.geom_type == "Polygon"
    assert polygon.area > 0
