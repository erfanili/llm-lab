"""TOML config files + `section.key=value` CLI overrides onto nested dataclasses."""

import tomllib
from dataclasses import fields


def load_config(cls, path: str, overrides: list[str]):
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for ov in overrides:
        key, val = ov.split("=", 1)
        section, name = key.split(".")
        try:
            parsed = tomllib.loads(f"v = {val}")["v"]
        except tomllib.TOMLDecodeError:
            parsed = val  # bare string, e.g. train.name=smoke
        raw.setdefault(section, {})[name] = parsed
    cfg = cls()
    for section, values in raw.items():
        obj = getattr(cfg, section)
        known = {f.name for f in fields(obj)}
        for k, v in values.items():
            if k not in known:
                raise KeyError(f"unknown config key {section}.{k}")
            setattr(obj, k, type(getattr(obj, k))(v))
    return cfg
