# doppel

A locally-served web app that scans a Google Drive photo library and surfaces
exact duplicates, near-duplicates, and visually similar images for human
review — with optional local-VLM adjudication and brand tagging via Ollama.
Nothing in Drive is ever modified (read-only scope); the output is a review
UI and an exported CSV of photos you marked for deletion.

Full specification: [SPEC.md](SPEC.md).

## Quickstart

```sh
make setup   # installs Python 3.12 + dependencies via uv
make run     # serves http://127.0.0.1:8000
```

Open **http://127.0.0.1:8000/setup** — the setup wizard walks you through:

1. **Google Drive** — upload an OAuth client JSON (Desktop app type, from
   [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
   with the Drive API enabled) and authorize in the browser. Read-only scope;
   credentials never leave your machine.
2. **Local LLM** — point at your Ollama server and pick an installed
   vision model (needs multi-image support: gemma3, qwen3-vl).
3. **Scan scope** — paste a Drive folder link to scan just that folder and
   its subfolders, or leave empty for the whole Drive.

Then run the stages from the dashboard in order: **sync → exact → near →
similar → adjudicate → brand**, review the groups, and export the CSV.

## Pipeline

| Tier | Meaning | Detection |
|---|---|---|
| exact | byte-identical files | Drive `md5Checksum` metadata |
| near | re-encoded / resized / lightly edited | perceptual hash (pHash + dHash) |
| similar | same scene, different shot | CLIP embeddings + sqlite-vec |
| vlm | borderline pairs ruled on by a local VLM | Ollama (multi-image vision model) |

All state lives in SQLite; every job is resumable and idempotent. Thresholds
live in [config.toml](config.toml). The brand candidate list lives in
[prompts/brand_v1.txt](prompts/brand_v1.txt) — edit it to match your wardrobe;
bump the filename version to re-run with a new prompt without losing prior
results.

## Development

```sh
make help    # all targets
make test    # pytest (fakes only — never hits the real Drive API)
make lint    # ruff check + format check
make scan    # CLI inventory sync (alternative to the dashboard button)
```
