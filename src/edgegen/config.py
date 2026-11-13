"""Configuration loading: every parameter lives in config/default.yaml (Table II of the paper)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def deep_update(base: dict, upd: Mapping) -> dict:
    """Return a copy of ``base`` recursively updated with ``upd``."""
    out = copy.deepcopy(base)
    for key, val in upd.items():
        if isinstance(val, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config; a file may ``inherit:`` another one and override some keys."""
    path = Path(path) if path is not None else CONFIG_DIR / "default.yaml"
    if not path.is_absolute() and not path.exists():
        path = PROJECT_ROOT / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    parent = cfg.pop("inherit", None)
    return deep_update(load_config(path.parent / parent), cfg) if parent else cfg


def with_overrides(cfg: dict, dotted: Mapping[str, Any]) -> dict:
    """Copy of ``cfg`` with overrides given as dotted keys, e.g. ``{"channel.bandwidth_mhz": 5}``."""
    nested: dict = {}
    for key, val in dotted.items():
        node = nested
        *parents, leaf = key.split(".")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = val
    return deep_update(cfg, nested)
