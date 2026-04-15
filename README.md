# `osmh3nx`

Hexagonal [H3](https://h3geo.org/) routing, drivesheds, and scalable network analysis with [OpenStreetMap](https://osm.org/) [networks](https://github.com/gboeing/osmnx).

[//]: # (Embedding badges: https://naereen.github.io/badges/)
[![Version](https://img.shields.io/badge/version-0.1.0b1-D12828.svg)](https://github.com/ccaoa/osmh3nx)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-E6BD29.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-0B6E4F.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/)
[![PyPI Latest Release](https://img.shields.io/pypi/v/osmh3nx.svg)](https://pypi.org/project/osmh3nx/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/osmh3nx.svg?label=PyPI%20downloads)](https://pypi.org/project/osmh3nx/)
<!-- [![CI - Tests](https://github.com/ccaoa/osmh3nx/actions/workflows/ci.yml/badge.svg)](https://github.com/ccaoa/osmh3nx/actions/workflows/ci.yml) -->

`osmh3nx` is a Python package for building reusable H3-based transportation networks from
OpenStreetMap drive graphs, then using those networks for routing, nearest-target assignment,
drivesheds, and (geo)dataframe-oriented batch analysis. 
The project is designed for users who want a scalable hexagonal abstraction over OSM routing
rather than using raw street-network nodes as the main analysis surface,
thus greatly enhancing computational efficiency.

## Building Blocks
[`osmh3nx`](https://github.com/ccaoa/osmh3nx) was created at [Child Care Aware® of America](https://www.childcareaware.org/) and is built upon:
- [`h3`](https://h3geo.org/), originally developed by [Uber](https://github.com/uber), provides the hexagonal global grid system that serves as the package's core routing and storage abstraction.
- [`osmnx`](https://osmnx.readthedocs.io/en/stable/) handles OpenStreetMap graph download, enrichment, and graph-to-GeoDataFrame conversion.
- [`networkx`](https://networkx.org/en/) powers the graph structure itself along with shortest-path and connectivity operations.
- [`GeoPandas`](https://geopandas.org/en/stable/), [`shapely`](https://shapely.readthedocs.io/), and [`pandas`](http://pandas.pydata.org/) provide the geometry, tabular, and dataframe-first workflow foundation used throughout the package outputs and batch interfaces.

## Inspiration

* Rabinowitz, N. 2023. "H3 Travel Times." https://observablehq.com/@nrabinowitz/h3-travel-times
* Malla, S. R. 2025. A landmark-based addressing framework for urban navigation using geospatial clustering and pathfinding algorithm. *Kathmandu University Journal of Science Engineering and Technology*, 19(1). https://doi.org/10.70530/kuset.v19i1.593
* Boeing, G. 2025. Modeling and Analyzing Urban Networks and Amenities with OSMnx. *Geographical Analysis* 57(4), 567-577. https://doi.org/10.1111/gean.70009

## Installation

Install from PyPI:

```bash
pip install osmh3nx
```

`osmh3nx` currently targets Python `3.11` and `3.12`.

## What the Package Does

At a high level, `osmh3nx`:

1. Downloads or reuses a drivable, directional OpenStreetMap network
2. Collapses that network into an H3-based travel graph
3. Preserves OSM-derived connectivity while operating on hexagons rather than raw street nodes
4. Exposes routing, nearest-target assignment, drivesheds, and dataframe-first batch tools

This package is most useful when you want scalable area-wide travel-time analysis
and are willing to work with a hex-based abstraction instead of exact turn-by-turn street routing.


## Typical Use Cases

- Build a reusable, directional, and abstracted drive network for a study area
- Find the nearest destination for many source points by travel time
- Generate drivesheds around many origin points
- Reuse one shared network context for clustered batch runs, even across large areas
- Store unique H3 cell geometry once and keep origin-to-cell relationships in a sidecar table
- Explore which origins cross a given H3 cell, or which cells belong to a given origin's driveshed

## Core Concepts

### H3-first network analysis

`osmh3nx` uses H3 cells as the main network abstraction. OSM is still the source of the road topology and travel-time evidence, but the downstream analysis surface is the hex grid.

### Calibrated travel cost

The package includes a reusable calibration layer so network-building settings do not need to be duplicated across every script. Current defaults are intentionally configurable and remain subject to future tuning.

### Dataframe-first outputs

Batch tools are built around `pandas` and `GeoPandas` outputs. The package is designed so batch results can be inspected directly in Python and also written cleanly to GIS-friendly formats.

## Main Public Functions

The top-level package currently exposes the main entry points you are most likely to use:

- `build_calibrated_h3_graph_from_osm` builds a calibrated H3 travel graph directly from an OSM drive graph.
- `build_calibrated_h3_graph_for_points` builds and packages a reusable batch graph context for one point set or group.
- `run_batch_od_routes` solves many origin-destination routes and returns tabular plus spatial route outputs.
- `run_batch_nearest_target_assignment` assigns each source point to its nearest target by H3-network travel cost.
- `run_batch_drivesheds` builds many drivesheds and returns unique H3 cell geometry plus origin-to-cell lookup rows. Grouped batch drivesheds rely on buffered convex hull geometries from the batch input points.
- `build_h3_driveshed_from_point` builds a single-origin driveshed from a point and returns detailed route-cell outputs.
- `select_driveshed_cells_from_lookup` reconstructs the unique cell geometry for one or more selected origins.
- `select_origins_for_driveshed_cell` returns all origin lookup rows whose drivesheds cross a given H3 cell.
- `dissolve_driveshed_cells_from_lookup` rebuilds dissolved driveshed polygons on demand from normalized cell storage.

## Quick Start

### Single-origin driveshed

```python
from shapely.geometry import Point

from osmh3nx import build_h3_driveshed_from_point

origin = Point(-79.94143, 37.27097)

result = build_h3_driveshed_from_point(
    origin,
    max_travel_minutes=20.0,
)

print(result.reachable_cells_gdf.head())
print(result.driveshed_gdf.head())
```

This returns a `DriveshedResult` object with:

- `reachable_cells_gdf`
- `reachable_edges_gdf`
- `driveshed_gdf`
- the origin cell ids used in routing
- the underlying H3 graph and optional OSM graph context

### Batch drivesheds from a GeoDataFrame

```python
import geopandas as gpd

from osmh3nx import run_batch_drivesheds

origins = gpd.GeoDataFrame(
    {"origin_id": ["a", "b"]},
    geometry=gpd.points_from_xy(
        [-79.94143, -79.94500],
        [37.27097, 37.27200],
    ),
    crs="EPSG:4326",
)

result = run_batch_drivesheds(
    origins,
    origin_id_col="origin_id",
    share_graph_for_all_origins=True,
    max_travel_minutes=20.0,
)

print(result.driveshed_cells_unique_gdf.head())
print(result.driveshed_cell_lookup_df.head())
```

`run_batch_drivesheds(...)` now uses a normalized storage model:

- `driveshed_cells_unique_gdf` stores unique H3 cell geometry once
- `driveshed_cell_lookup_df` stores origin-to-cell relationships as a non-spatial table

This is much more scalable than writing duplicated polygon geometry for every origin-cell pair.

### Batch OD routing / nearest-target assignment

If your workflow is origin-destination routing rather than drivesheds, use the dataframe-oriented batch helpers:

- `run_batch_od_routes(...)`
- `run_batch_nearest_target_assignment(...)`

These functions accept `pandas` or `GeoPandas` inputs and return structured batch result objects with routes, route hexes, search polygons, and reusable graph contexts.

## Working With Normalized Driveshed Outputs

One of the main downstream patterns in `osmh3nx` is:

- write unique cell geometry to a GeoPackage
- write the lookup table to a sidecar table
- reconstruct subsets on demand rather than storing massive duplicated geometry layers

### Get all cells for one origin

```python
from osmh3nx import select_driveshed_cells_from_lookup

cells_for_origin = select_driveshed_cells_from_lookup(
    unique_cells_gdf,
    cell_lookup_df,
    origin_ids=["1412817"],
)
```

### Get all origins that cross one H3 cell

```python
from osmh3nx import select_origins_for_driveshed_cell

origins_for_cell = select_origins_for_driveshed_cell(
    cell_lookup_df,
    h3_cell="8a2a8a905aeffff",
)
```

### Rebuild a dissolved polygon on demand

```python
from osmh3nx import dissolve_driveshed_cells_from_lookup

dissolved = dissolve_driveshed_cells_from_lookup(
    unique_cells_gdf,
    cell_lookup_df,
    origin_ids=["1412817"],
)
```

This allows large batch runs to stay compact while still supporting targeted visualization later.

## Output Formats

Current common output pattern:

- spatial layers to GeoPackage
- lookup tables to sidecar files

The lookup sidecar supports:

- `parquet` by default at the module level for scale and performance
- `csv` when human readability or quick GIS joins are more important

## Caching And Temporary Data

`osmh3nx` can reuse cached OSM graph downloads where configured,
and downstream callers can pass a custom `cache_dir` to the relevant network/driveshed/batch functions when they want cache files written somewhere specific.
At the package level, the default cache location comes from the user's platform cache directory via `platformdirs` rather than from the downstream working directory or this repo.

The package test suite is intentionally offline-first and avoids persistent scratch data by using tiny synthetic fixtures and pytest temporary directories.

## GIS Workflow Notes

The package is designed to work well with GIS GUI tools:

- unique H3 cell layers can be rendered directly from GeoPackage
- lookup sidecars can be joined as non-spatial tables
- helper functions are available for reconstructing subsets in Python
- the same lookup logic can be replicated in GIS through normal joins and filters

## Command Line Entry Point

The package installs a console script:

```bash
osmh3nx-driveshed
```

This maps to the package driveshed CLI entry point. The repo's `scripts/` directory also contains larger workflow scripts used during development, calibration, and batch testing.

## Limitations And Expectations

- This package is not a replacement for exact street-node routing when exact turn-level path fidelity is the main objective.
- H3 resolution choice matters and changes both runtime and route behavior.
- Travel-time realism depends on calibration settings and study-area context.
- Large batch runs can still be computationally heavy even with shared-graph reuse.
- Some development scripts in `scripts/` are research-oriented and broader than the minimal package API surface.

## Development

Run tests:

```bash
python -m pytest tests -ra
```

Check formatting:

```bash
black . --check --target-version py311
```

Install in editable mode:

```bash
python -m pip install -e .[development]
```

## License

This project is licensed under the BSD 3-Clause License. See [LICENSE](LICENSE).
