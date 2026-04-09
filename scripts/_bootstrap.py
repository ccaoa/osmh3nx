from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def src_root() -> Path:
    return repo_root() / "src"


def ensure_src_on_path() -> Path:
    root = src_root()
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root
