from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from osmh3nx.io import write_table_sidecar


def test_write_table_sidecar_csv(tmp_path: Path) -> None:
    table = pd.DataFrame({"id": [1, 2], "value": ["a", "b"]})
    output_path = tmp_path / "table.csv"

    written_path = write_table_sidecar(table, output_path, table_format="csv")

    assert written_path == output_path
    assert output_path.exists()
    assert pd.read_csv(output_path).equals(table)


def test_write_table_sidecar_default_uses_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = pd.DataFrame({"id": [1], "value": ["x"]})
    output_path = tmp_path / "table.parquet"
    called: dict[str, object] = {}

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        called["path"] = path
        called["index"] = index
        path.write_text("stub parquet output", encoding="utf-8")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    written_path = write_table_sidecar(table, output_path)

    assert written_path == output_path
    assert called["path"] == output_path
    assert called["index"] is False
    assert output_path.exists()


def test_write_table_sidecar_parquet_raises_clear_error_without_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = pd.DataFrame({"id": [1]})
    output_path = tmp_path / "table.parquet"

    def fake_to_parquet(self: pd.DataFrame, path: Path, index: bool = False) -> None:
        raise ImportError("missing parquet backend")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)

    with pytest.raises(ImportError, match="optional parquet engine"):
        write_table_sidecar(table, output_path, table_format="parquet")
