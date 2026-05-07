# `osmh3nx` Changelog 
[comment]: # (Website for propper changelog documentation: [https://keepachangelog.com/en/1.1.0/])

This package's releases adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased <To become the release notes for the next version>
[//]: # (tagging git releases https://stackoverflow.com/questions/18216991/create-a-tag-in-a-github-repository)
* *None*

### 0.1.0b1
**15 Apr 2026**

Released with [Pull Request #2](https://github.com/ccaoa/osmh3nx/pull/2)
* First beta prerelease for the upcoming `0.1.0` initial release.
* Established the core `osmh3nx` package structure under [`src/`](src) with PyPI-oriented metadata, BSD 3-Clause [licensing](LICENSE), version synchronization, and release workflow scaffolding.
* Added reusable OSM-to-H3 network construction that downloads or reuses OpenStreetMap drive graphs, enriches travel-time attributes, and builds calibrated H3 travel graphs for routing workflows.
* Added primary analysis workflows for H3 routing, nearest-target assignment, and single-origin drivesheds, including a command-line driveshed entry point.
* Added dataframe-first batch operators for OD routing, nearest-target assignment, and drivesheds, with reusable shared-graph contexts for clustered batch runs.
* Added normalized batch driveshed storage that writes unique H3 cell geometry once and keeps origin-to-cell relationships in tabular lookup outputs for scalable downstream analysis.
* Added lookup and reconstruction helpers for downstream workflows, including selecting all cells for one origin, selecting all origins crossing one H3 cell, and rebuilding dissolved driveshed polygons on demand.
* Added foundational development infrastructure including GitHub Actions for CI, package checks, publishing/release management, an offline-first pytest suite, and repo scripts for calibration and batch testing.
#### Issues Closed 
n=0
* *None*
#### Deprecated
* *None*
