# CLAUDE.md

## Project

doppel — a locally-served web app that scans a Google Drive photo library and
surfaces exact duplicates, near-duplicates, and visually similar images for
human review. Full specification and phased build plan: `SPEC.md`.

## Stack

- Python 3.12, `uv` for env/dependency management
- FastAPI + Jinja2 + htmx (server-rendered UI, minimal JS)
- SQLite (stdlib `sqlite3`) + `sqlite-vec` for embeddings
- Pillow + `imagehash` (perceptual hashing)
- `open_clip_torch` on MPS (image embeddings)
- `google-api-python-client` + `google-auth-oauthlib` (Drive)
- Ollama over local HTTP via the `ollama` package (vision tasks, Phases 6–7)

## Commands

Use Makefile targets, never raw commands. `make help` lists all targets.
Core: `make setup`, `make run`, `make test`, `make lint`, `make scan`.

## Rules

- Drive scope is `drive.readonly`. Never request a write scope. v1 never
  modifies anything in Drive.
- All image bytes go through the `ImageFetcher` abstraction (SPEC.md):
  size-parameterized thumbnails (default 512px) cached to `cache/`.
  Original-size fetch is used by Phase 7 brand tagging only — never for
  duplicate detection.
- Every VLM call goes through `OllamaClient`: JSON-schema-forced output,
  versioned prompt files from `prompts/`, model + prompt_version stored
  with each result.
- Detection thresholds (Hamming distance, cosine similarity) live in
  `config.toml`, never hardcoded.
- `credentials.json`, `token.json`, `cache/`, `*.db` are gitignored.
  Never commit, print, or log credentials or tokens.
- Bind the web server to 127.0.0.1 only.
- All persistent state lives in SQLite. No JSON/pickle state files.
- Jobs must be idempotent and resumable: skip photos whose stage output
  (hash, embedding) already exists.
- Work through SPEC.md phases in order. Advance to the next phase only
  when the current phase's acceptance criteria pass under
  self-verification: `make test` and `make lint` green, the server boots,
  and the criteria are exercised against fixtures and the fake Drive
  client — never the real Drive API.
- Stop and ask the human only for what cannot be self-verified: the OAuth
  consent flow, judgments about results on the real photo library, and
  threshold tuning.
- Do not invent work. Record bugs, optimization ideas, and UI/UX
  improvements in BACKLOG.md instead of acting on them, unless they block
  the current phase's acceptance criteria. Stop when the instructed phase
  range is complete.
- Commit at every phase boundary.
- `ruff` for lint + format, `pytest` for tests, type hints on all public
  functions. Tests must not hit the real Drive API — use the fake client.
- Do not add Claude as a co-author on git commits.
