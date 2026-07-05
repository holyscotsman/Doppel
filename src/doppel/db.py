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
  parent_id      TEXT,     -- immediate Drive parent folder id
  folder_path    TEXT,     -- resolved 'grandparent / parent / folder' for display
  status         TEXT NOT NULL DEFAULT 'active'  -- active | missing | trashed
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
-- the PK leads with group_id, so a lookup by photo_id (the move-to-trash
-- "is this photo's group reviewed?" check) would otherwise scan the table
CREATE INDEX IF NOT EXISTS idx_group_members_photo ON group_members(photo_id);

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

CREATE TABLE IF NOT EXISTS vlm_results (
  id             INTEGER PRIMARY KEY,
  task           TEXT NOT NULL,        -- adjudicate
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
    # WAL allows one writer at a time; without a busy timeout a second writer
    # (e.g. /trash on a request thread while a background scan writes) fails
    # immediately with "database is locked". Wait a few seconds instead.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created. CREATE TABLE
    IF NOT EXISTS won't add columns to an existing table, so do it here."""
    have = {row["name"] for row in conn.execute("PRAGMA table_info(photos)")}
    for column in ("parent_id TEXT", "folder_path TEXT"):
        name = column.split()[0]
        if name not in have:
            conn.execute(f"ALTER TABLE photos ADD COLUMN {column}")
    conn.commit()


def get_meta(
    conn: sqlite3.Connection, key: str, default: str | None = None
) -> str | None:
    """Read a value from the small key/value meta table."""
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Write a value to the meta table."""
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()


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
