"""Minimal Sphinx config for the top-level share-rl wiki."""

from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

sys.path.insert(0, os.fspath(ROOT))
sys.path.insert(0, os.fspath(SRC))

project = "share-rl"
author = "share-rl contributors"

extensions: list[str] = []
templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "alabaster"
html_static_path = ["_static"]
