import sqlite3

import pytest

from doppel.config import Config, OllamaConfig
from doppel.db import connect


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    return connect(tmp_path / "test.db")


@pytest.fixture
def config(tmp_path) -> Config:
    return Config(
        thumb_size=512,
        near_hamming_max=8,
        dhash_confirm_max=10,
        similar_cosine_min=0.92,
        color_variant_min_delta=0.25,
        clip_model="ViT-B-32/laion2b_s34b_b79k",
        db_path=tmp_path / "test.db",
        cache_dir=tmp_path / "cache",
        ollama=OllamaConfig(
            host="http://127.0.0.1:11434",
            model="test-model",
            adjudicate_band_min=0.85,
            brand_review_max_confidence=0.6,
        ),
    )
