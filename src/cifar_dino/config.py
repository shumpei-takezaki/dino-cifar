"""Plain-YAML config: a method config names its parent via ``_base_`` and is deep-merged over it."""

from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("_base_", None)
    return deep_merge(load_config(path.parent / base), cfg) if base else cfg
