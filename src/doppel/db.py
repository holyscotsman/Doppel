"""SQLite schema and connection helpers. All persistent state lives here."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
  id             INTEGER PRIMARY KEY,
  drive_id       TEXT UNIQUE NOT NULL,
  name           TEXT NOT NULL,
  mime_type      TEXT NOT NULL,
  size           INTEGER,
  md5            TEXT,
  width          INTEGER,
  height         INTEGER,
  created_time   TEXT,
  modified_time  TEXT,
  thumbnail_link TEXT,     -- Drive thumbnailLink; expires, refreshed on demand
  thumb_path     TEXT,     -- local cache path, NULL until fetched
  phash          TEXT,     -- 16-char hex, NULL until computed
  dhash          TEXT,
  status         TEXT NOT NULL DEFAULT 'active'  -- active | missing
);
CREATE INDEX IF NOT EXISTS idx_photos_md5 ON photos(md5);

CREATE TABLE IF NOT EXISTS groups (
  id            INTEGER PRIMARY KEY,
  tier          TEXT NOT NULL,   -- exact | near | similar | vlm
  color_variant INTEGER NOT NULL DEFAULT 0,  -- structure matches, color differs
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS group_members (
  group_id INTEGER NOT NULL REFERENCES groups(id),
  photo_id INTEGER NOT NULL REFERENCES photos(id),
  score    REAL,                 -- hamming dist or cosine sim vs group anchor
  PRIMARY KEY (group_id, photo_id)
);

CREATE TABLE IF NOT EXISTS decisions (
  photo_id   INTEGER PRIMARY KEY REFERENCES photos(id),
  action     TEXT NOT NULL,      -- keep | trash
  decided_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scans (
  id          INTEGER PRIMARY KEY,
  stage       TEXT NOT NULL,     -- sync | exact | near | similar | adjudicate
  status      TEXT NOT NULL,     -- running | done | failed
  processed   INTEGER DEFAULT 0,
  total       INTEGER,
  started_at  TEXT,
  finished_at TEXT,
  error       TEXT
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
  photo_id   INTEGER NOT NULL REFERENCES photos(id),
  kind       TEXT NOT NULL,            -- 'brand'
  value      TEXT NOT NULL,
  confidence REAL,
  source     TEXT NOT NULL,            -- vlm | human
  created_at TEXT NOT NULL,
  PRIMARY KEY (photo_id, kind)
);

CREATE TABLE IF NOT EXISTS vlm_results (
  id             INTEGER PRIMARY KEY,
  task           TEXT NOT NULL,        -- adjudicate | brand
  photo_id       INTEGER NOT NULL REFERENCES photos(id),
  photo_id_b     INTEGER REFERENCES photos(id),  -- adjudicate only
  model          TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  response       TEXT NOT NULL,        -- raw JSON from the model
  verdict        TEXT,
  confidence     REAL,
  created_at     TEXT NOT NULL
);
"""


def connect(db_path: Path | str) -> sqlite3.Connection:
    """Open a connection with sane defaults and the schema applied.

    check_same_thread=False: FastAPI resolves sync dependencies in a
    threadpool thread while async routes run on the event loop; each
    connection is still used by one request at a time.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.executescript(SCHEMA)
    return conn


def ensure_vec_schema(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension and create the embeddings virtual table.

    Called by the similar stage (and anything reading vectors) rather than
    connect(), so the core schema works without the extension present.
    """
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
          photo_id  INTEGER PRIMARY KEY,
          embedding FLOAT[512] distance_metric=cosine
        )
        """
    )
