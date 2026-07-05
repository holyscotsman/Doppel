"""Configuration loading. All tunables live in config.toml, never in code."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OllamaConfig:
    host: str
    model: str
    adjudicate_band_min: float


@dataclass(frozen=True)
class PerfConfig:
    """How hard to push the machine during the deeper scans. Worker counts and
    batch sizes are never hardcoded in the stages — they come from here so the
    [perf] section in config.toml can tune them to the host. Defaults suit a
    modern multi-core Mac; load_config fills worker counts from os.cpu_count()."""

    fetch_workers: int = 8  # parallel thumbnail fetches (I/O bound)
    hash_workers: int = 8  # parallel fetch + pHash/dHash in the near stage
    embed_fetch_workers: int = 8  # prefetch threads feeding the CLIP batcher
    clip_batch: int = 32  # images per embedder.embed() pass on the GPU
    db_batch: int = 100  # rows per sqlite transaction from the single writer
    adjudicate_workers: int = 3  # fetch + VLM threads (Ollama is one server)
    queue_maxsize: int = 40  # bounded worker→writer handoff, for backpressure


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
    perf: PerfConfig = field(default_factory=PerfConfig)
    # when a stage resumes after an interruption, re-do this many of the most
    # recently finished photos — cheap insurance against a half-written result
    # or a truncated fetch right at the point of the crash.
    resume_overlap: int = 3
    # review preselect: within a duplicate group, prefer trashing the copy whose
    # folder path contains sort_folder_keyword (e.g. a "To Sort" inbox), keeping
    # the copy filed in a real folder. On a tie (all or none in such a folder)
    # the largest file is kept. Turn off to always just keep the largest.
    prefer_trash_sort: bool = True
    sort_folder_keyword: str = "sort"


def load_config(path: Path | str = "config.toml") -> Config:
    """Load configuration from a TOML file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    ollama = raw["ollama"]
    perf = raw.get("perf", {})
    cpu = os.cpu_count() or 4
    # Thumbnail fetching is network-I/O bound, not CPU bound: a worker spends
    # almost all its time waiting on an HTTPS round trip, so useful concurrency
    # far exceeds the core count. Default the fetch/hash/prefetch pools to 4x
    # cores (capped at 32 to stay polite to the Drive thumbnail CDN, which the
    # fetcher already backs off from on 429). This is only safe now that each
    # worker holds its own connection — before, extra threads just corrupted a
    # shared socket and crashed the scan. Raise these in [perf] if your link
    # has headroom and you are not being rate-limited.
    io_workers = min(32, cpu * 4)
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
        ),
        perf=PerfConfig(
            fetch_workers=perf.get("fetch_workers", io_workers),
            hash_workers=perf.get("hash_workers", io_workers),
            embed_fetch_workers=perf.get("embed_fetch_workers", io_workers),
            clip_batch=perf.get("clip_batch", 32),
            db_batch=perf.get("db_batch", 100),
            adjudicate_workers=perf.get("adjudicate_workers", 3),
            queue_maxsize=perf.get("queue_maxsize", max(4 * cpu, 2 * io_workers)),
        ),
        resume_overlap=raw.get("resume_overlap", 3),
        prefer_trash_sort=raw.get("prefer_trash_sort", True),
        sort_folder_keyword=raw.get("sort_folder_keyword", "sort"),
    )


def set_config_value(
    path: Path | str, key: str, value: str, section: str | None = None
) -> None:
    """Rewrite one `key = "value"` line in config.toml, preserving comments
    and formatting. The setup wizard's only write path into configuration.

    Values are JSON-escaped, which is valid TOML basic-string escaping —
    quotes, backslashes, and newlines can never corrupt the file or inject
    additional keys.
    """
    import json
    import re

    path = Path(path)
    lines = path.read_text().splitlines(keepends=True)
    current_section: str | None = None
    # leading whitespace before a key is valid TOML — match it too, or an
    # indented existing key would get a duplicate appended
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    quoted = json.dumps(value)
    new_line = f"{key} = {quoted}\n"
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("["):
            current_section = stripped.strip("[]")
        if current_section == section and pattern.match(line):
            # keep an inline comment if present
            comment = line.split("#", 1)
            suffix = f"  #{comment[1]}" if len(comment) > 1 else "\n"
            lines[i] = f"{key} = {quoted}" + (
                suffix if suffix.startswith("  #") else "\n"
            )
            path.write_text("".join(lines))
            return
    # key absent: insert it in the right place
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
