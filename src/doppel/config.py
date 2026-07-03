"""Configuration loading. All tunables live in config.toml, never in code."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str
    adjudicate_band_min: float


@dataclass(frozen=True)
class Config:
    thumb_size: int
    near_hamming_max: int
    dhash_confirm_max: int
    similar_cosine_min: float
    color_variant_min_delta: float
    clip_model: str
    db_path: Path
    cache_dir: Path
    ollama: OllamaConfig


def load_config(path: Path | str = "config.toml") -> Config:
    """Load configuration from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    ollama = raw["ollama"]
    return Config(
        thumb_size=raw["thumb_size"],
        near_hamming_max=raw["near_hamming_max"],
        dhash_confirm_max=raw["dhash_confirm_max"],
        similar_cosine_min=raw["similar_cosine_min"],
        color_variant_min_delta=raw["color_variant_min_delta"],
        clip_model=raw["clip_model"],
        db_path=Path(raw["db_path"]),
        cache_dir=Path(raw["cache_dir"]),
        ollama=OllamaConfig(
            host=ollama["host"],
            model=ollama["model"],
            adjudicate_band_min=ollama["adjudicate_band_min"],
        ),
    )
