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
    brand_review_max_confidence: float


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
    drive_folder_id: str  # "" = scan the entire Drive
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
        drive_folder_id=raw.get("drive_folder_id", ""),
        ollama=OllamaConfig(
            host=ollama["host"],
            model=ollama["model"],
            adjudicate_band_min=ollama["adjudicate_band_min"],
            brand_review_max_confidence=ollama["brand_review_max_confidence"],
        ),
    )


def set_config_value(
    path: Path | str, key: str, value: str, section: str | None = None
) -> None:
    """Rewrite one `key = "value"` line in config.toml, preserving comments
    and formatting. The setup wizard's only write path into configuration.
    """
    import re

    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    current_section: str | None = None
    pattern = re.compile(rf"^{re.escape(key)}\s*=")
    new_line = f'{key} = "{value}"\n'
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            current_section = stripped.strip("[]")
        if current_section == section and pattern.match(line):
            # keep an inline comment if present
            comment = line.split("#", 1)
            suffix = f"  #{comment[1]}" if len(comment) > 1 else "\n"
            lines[i] = f'{key} = "{value}"' + (
                suffix if suffix.startswith("  #") else "\n"
            )
            path.write_text("".join(lines))
            return
    # key absent: append it to the right place
    if section is None:
        # insert before the first section header, or at the end
        insert_at = next(
            (i for i, ln in enumerate(lines) if ln.strip().startswith("[")),
            len(lines),
        )
        lines.insert(insert_at, new_line)
    else:
        header = f"[{section}]"
        try:
            start = next(i for i, ln in enumerate(lines) if ln.strip() == header)
        except StopIteration:
            lines.append(f"\n{header}\n")
            start = len(lines) - 1
        lines.insert(start + 1, new_line)
    path.write_text("".join(lines))
