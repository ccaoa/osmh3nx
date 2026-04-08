# `osmh3nx`

Hexagonal [H3](https://h3geo.org/) routing, drivesheds, and scalable network analysis with [OpenStreetMap](https://osm.org/) [networks](https://github.com/gboeing/osmnx).

[//]: # (Embedding badges: https://naereen.github.io/badges/)
[![Version](https://img.shields.io/badge/version-0.1.0-D12828.svg)](https://github.com/ccaoa/osmh3nx)
[![Python versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-E6BD29.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/)

`osmh3nx` is a Python package for building reusable H3-based transportation networks from
OpenStreetMap drive graphs, then using those networks for routing, nearest-target assignment,
drivesheds, and dataframe-oriented batch analysis. The project is designed for users who want a
scalable hex-based abstraction over OSM routing rather than using raw street-network nodes as the
main analysis surface.

Inspired by 

* Rabinowitz, N. 2023. "H3 Travel Times." https://observablehq.com/@nrabinowitz/h3-travel-times
* Malla, S. R. 2025. A landmark-based addressing framework for urban navigation using geospatial clustering and pathfinding algorithm. *Kathmandu University Journal of Science Engineering and Technology*, 19(1). https://doi.org/10.70530/kuset.v19i1.593

