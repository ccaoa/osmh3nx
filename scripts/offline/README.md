# Offline Folder Guide

This folder contains tools and artifacts for offline routing prep and map review.

## What Is In Here

- `offline_prep.py`
  - Builds offline artifacts from `osm_scale_calibration.csv`.
  - Saves one folder per OD pair with:
    - `osm_drive.graphml` (OSM drive graph for that pair area)
    - `h3_drive_res*.pkl` (H3 travel graphs by resolution)
    - `manifest.json`
  - Also writes `offline_bundle_manifest.json` at bundle root.

- `export_pair_routes_qgis.py`
  - Reads one `pair_*` folder and exports:
    - OSM truth route
    - H3 route lines by selected resolutions
    - H3 route hex polygons by selected resolutions
  - Output is a QGIS-ready GPKG.

- `pkl_to_gpkg.py`
  - Converts one `h3_drive_res*.pkl` graph to a QGIS-ready GPKG with:
    - `h3_nodes_poly`
    - `h3_nodes_centroid`
    - `h3_edges_centroid_line`

- `offline_bundle_smoke/`
  - Small test bundle from smoke runs.

- `osmh3nx_offline_bundle/`
  - Main offline bundle(s) generated from full/partial runs.


## When To Use This Folder

Use `offline/` when you want to:

- Cache OSM + H3 graphs ahead of travel/no internet.
- Inspect one pair at a time in QGIS.
- Debug route geometry and snapping without redownloading data.


## When To Use Root Calibration Script Instead

Use root script `h3_osm_calibration.py` when your goal is calibration metrics across many cities/resolutions.

That script is the canonical calibration pipeline because it computes and writes:

- OSM baseline route metrics
- H3 route metrics
- Error fields (`time_error_*`, `distance_error_*`)
- Consolidated outputs:
  - `h3_osm_calibration_vintage{vintage}.gpkg`
  - `h3_osm_calibration_vintage{vintage}_metrics.csv`

In short:

- `offline/` scripts: prep/cache/export helpers
- root `h3_osm_calibration.py`: primary model validation and scale calibration run


## Typical Commands (Run From Repo Root)

Prepare offline bundle:

```powershell
.\venv\Scripts\python.exe .\offline\offline_prep.py `
  --csv .\osm_scale_calibration.csv `
  --out-dir .\offline\osmh3nx_offline_bundle `
  --resolutions 7,8,9,10 `
  --sample-miles 0.1
```

Export one pair route review GPKG:

```powershell
.\venv\Scripts\python.exe .\offline\export_pair_routes_qgis.py `
  --pair-dir .\offline\osmh3nx_offline_bundle\pair_00001 `
  --out-gpkg .\offline\osmh3nx_offline_bundle\pair_00001\pair_00001_routes_review.gpkg `
  --resolutions 7,8,9,10
```

Convert one H3 pickle graph to QGIS layers:

```powershell
.\venv\Scripts\python.exe .\offline\pkl_to_gpkg.py `
  --pkl .\offline\osmh3nx_offline_bundle\pair_00001\h3_drive_res8.pkl `
  --out-gpkg .\offline\osmh3nx_offline_bundle\pair_00001\h3_drive_res8_qgis.gpkg
```

Run full calibration (preferred for metrics):

```powershell
.\venv\Scripts\python.exe .\h3_osm_calibration.py
```
