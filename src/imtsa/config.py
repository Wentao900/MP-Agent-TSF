from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    parent = cfg.get("extends")
    if not parent:
        return cfg
    parent_path = (path.parent / parent).resolve() if not Path(parent).is_absolute() else Path(parent)
    parent_cfg = load_config(parent_path)
    cfg = dict(cfg)
    cfg.pop("extends", None)
    return _merge_dict(parent_cfg, cfg)
