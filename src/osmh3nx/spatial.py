from __future__ import annotations

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from . import driveshed as dshed
from . import network_osm as onetx

GEOMETRY_COLUMN: str = "geometry"
DEFAULT_BUFFERED_CONVEX_HULL_BUFFER_MILES: float = dshed.DEFAULT_SHARED_GRAPH_BUFFER_MILES


def _parse_version_parts(version_text: str) -> tuple[int, int]:
    """
    Convert a package version string into a comparable major/minor tuple.
    Examples such as '1.1.1' and '1.0.0+dev' both resolve safely.
    """
    numeric_parts: list[int] = []
    for part in str(version_text).split("."):
        digits = ""
        for char in part:
            if char.isdigit():
                digits += char
            else:
                break
        numeric_parts.append(int(digits) if digits else 0)

    while len(numeric_parts) < 2:
        numeric_parts.append(0)

    return numeric_parts[0], numeric_parts[1]


def create_convex_hull(in_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Create a single polygon GeoDataFrame representing the convex hull around all
    valid geometries in the input selection.

    Parameters:
        in_gdf: Input GeoDataFrame containing the selected features.

    Returns:
        A one-row GeoDataFrame containing the convex hull polygon with the same
        CRS as the input.

    Notes:
        - This is narrower-fitting than a bounding box but remains convex.
        - GeoPandas/Shapely union APIs have varied across versions. This helper
          checks the installed GeoPandas version and falls back to older access
          patterns when needed so callers do not need to manage stack
          differences.
    """
    if not isinstance(in_gdf, gpd.GeoDataFrame):
        raise TypeError("`in_gdf` must be a GeoDataFrame.")

    if GEOMETRY_COLUMN not in in_gdf.columns:
        raise ValueError("The input GeoDataFrame must contain a geometry column.")

    if in_gdf.empty:
        raise ValueError("Cannot create a convex hull from an empty GeoDataFrame.")

    valid_geometries = in_gdf.geometry.dropna()
    valid_geometries = valid_geometries[~valid_geometries.is_empty]

    if valid_geometries.empty:
        raise ValueError(
            "Cannot create a convex hull because no non-null, non-empty geometries were found."
        )

    gpd_version = _parse_version_parts(getattr(gpd, "__version__", "0.0"))

    # Prefer the newer union_all path when it is known to be available, but keep
    # a capability-based fallback so this helper remains resilient to API drift.
    if gpd_version >= (1, 0) and hasattr(valid_geometries, "union_all"):
        merged_geometry = valid_geometries.union_all()
    else:
        merged_geometry = valid_geometries.unary_union

    hull_geometry = merged_geometry.convex_hull
    out_gdf = gpd.GeoDataFrame(
        [{GEOMETRY_COLUMN: hull_geometry}],
        geometry=GEOMETRY_COLUMN,
    )

    if in_gdf.crs:
        out_gdf = out_gdf.set_crs(in_gdf.crs)

    return out_gdf


def create_buffered_convex_hull(
    in_gdf: gpd.GeoDataFrame,
    *,
    buffer_miles: float = DEFAULT_BUFFERED_CONVEX_HULL_BUFFER_MILES,
) -> gpd.GeoDataFrame:
    """
    Create a convex hull around all valid geometries, then buffer it outward.

    Parameters:
        in_gdf: Input GeoDataFrame containing the selected features.
        buffer_miles: Outward buffer distance in miles. Defaults to the shared
            driveshed-group buffer derived from `driveshed.DEFAULT_SEARCH_BUFFER_FACTOR`.

    Returns:
        A one-row GeoDataFrame containing the buffered convex hull polygon in the
        same CRS as the input.
    """
    if buffer_miles < 0:
        raise ValueError("buffer_miles must be >= 0.")

    hull_gdf = create_convex_hull(in_gdf)
    if buffer_miles == 0:
        return hull_gdf
    if hull_gdf.crs is None:
        raise ValueError("A CRS is required to buffer the convex hull.")

    proj_crs = hull_gdf.estimate_utm_crs() or "EPSG:3857"
    hull_proj = hull_gdf.to_crs(proj_crs)
    buffered_geom: BaseGeometry = hull_proj.geometry.iloc[0].buffer(
        onetx.miles_to_meters(buffer_miles)
    )
    return gpd.GeoDataFrame(
        [{GEOMETRY_COLUMN: buffered_geom}],
        geometry=GEOMETRY_COLUMN,
        crs=proj_crs,
    ).to_crs(hull_gdf.crs)
