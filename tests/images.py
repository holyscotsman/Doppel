"""Deterministic image fixtures for the detection-stage tests.

Distances validated empirically: resized/recompressed copies of the
structured image are within near-tier thresholds (pHash 0, dHash <= 2);
the R/B-swapped colorway is at pHash 8 / dHash 7 (inside the default radius)
with an HSV histogram distance ~0.69 (above color_variant_min_delta); the
unrelated image is far outside (pHash >= 40).
"""

from __future__ import annotations

import io

from PIL import Image, ImageDraw


def structured_image() -> Image.Image:
    """A deterministic scene: gradient + shapes."""
    img = Image.new("RGB", (800, 600))
    px = img.load()
    for y in range(600):
        for x in range(800):
            px[x, y] = (30 + x * 140 // 800, 60 + y * 120 // 600, 160)
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 60, 280, 260], fill=(240, 220, 60))
    draw.rectangle([350, 300, 750, 560], fill=(40, 120, 60))
    draw.polygon([(400, 80), (560, 240), (300, 260)], fill=(200, 60, 40))
    draw.line([(0, 420), (800, 380)], fill=(250, 250, 250), width=12)
    return img


def unrelated_image() -> Image.Image:
    """Structurally different scene: vertical bars."""
    img = Image.new("RGB", (800, 600), (250, 250, 245))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        x = 30 + i * 64
        draw.rectangle([x, 40, x + 40, 560], fill=(10 + i * 18, 10, 40))
    return img


def color_variant(img: Image.Image) -> Image.Image:
    """Same structure, different colorway: swap R and B channels."""
    r, g, b = img.split()
    return Image.merge("RGB", (b, g, r))


def as_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
