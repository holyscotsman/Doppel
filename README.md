# doppel

A locally-served web app that scans a Google Drive photo library and surfaces
exact duplicates, near-duplicates, and visually similar images for human
review — with optional local-VLM adjudication of the borderline pairs via
Ollama. Nothing in Drive is ever modified (read-only scope); the output is a
review UI and an exported CSV of photos you marked for deletion.

**Landing page:** https://holyscotsman.github.io/Doppel/ &nbsp;·&nbsp;
Full specification: [SPEC.md](SPEC.md).

> doppel runs on your own machine — it needs your local Ollama, your compute,
> and your files. The site above is a front door; you connect Drive and Ollama
> in the local `/setup` wizard after `make run`, not on the web page.

## Quickstart

```sh
make setup   # installs Python 3.12 + dependencies via uv
make run     # serves http://127.0.0.1:8000
```

`make run` opens **http://127.0.0.1:8000/setup** — a guided wizard walks you
through:

1. **Google Drive** — a step-by-step guide to create a Google service-account
   key (no OAuth consent screen), drop it in, and share your photos folder with
   the account's email. Read-only scope; credentials never leave your machine.
2. **Local LLM** — point at your Ollama server; the model dropdown lists only
   installed vision models that can compare an image pair (gemma3, qwen3-vl).
3. **Scan scope** — browse your Drive and pick a folder to scan (its subfolders
   are included).

Then hit **Run full scan** on the dashboard (chains find → exact → near →
similar), optionally run the **AI double-check** (adjudicate), review the
groups, and export the CSV.

## Pipeline

| Tier | Meaning | Detection |
|---|---|---|
| exact | byte-identical files | Drive `md5Checksum` metadata |
| near | re-encoded / resized / lightly edited | perceptual hash (pHash + dHash) |
| similar | same scene, different shot | CLIP embeddings + sqlite-vec |
| vlm | borderline pairs ruled on by a local VLM | Ollama (multi-image vision model) |

All state lives in SQLite; every job is resumable and idempotent. Thresholds
live in [config.toml](config.toml), and the adjudication prompt is a versioned
file in [prompts/](prompts/) (bump the filename version to re-adjudicate
without losing prior results).

## Development

```sh
make help    # all targets
make test    # pytest (fakes only — never hits the real Drive API)
make lint    # ruff check + format check
make scan    # CLI inventory sync (alternative to the dashboard button)
```
