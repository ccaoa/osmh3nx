from __future__ import annotations

import pathlib
import tomllib

import osmh3nx
from osmh3nx import __version__ as package_init_version


def test_version_consistency() -> None:
    pyproject_version = tomllib.loads(pathlib.Path("pyproject.toml").read_text())[
        "project"
    ]["version"]
    version_file_ns: dict[str, str] = {}
    exec(pathlib.Path("src/osmh3nx/__version__.py").read_text(), version_file_ns)

    assert osmh3nx.__version__ == pyproject_version
    assert package_init_version == pyproject_version
    assert version_file_ns["__version__"] == pyproject_version


def test_top_level_exports_exist() -> None:
    assert callable(osmh3nx.build_h3_driveshed_from_point)
    assert callable(osmh3nx.create_convex_hull)
    assert callable(osmh3nx.select_driveshed_cells_from_lookup)
    assert callable(osmh3nx.select_origins_for_driveshed_cell)
