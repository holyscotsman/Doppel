# BACKLOG

Ideas and follow-ups recorded (not acted on) per CLAUDE.md. Not blocking any
phase's acceptance criteria.

## Scan robustness (follow-ups from the thread-safety/speed fix)

- **Credential refresh under high concurrency.** All per-thread `AuthorizedSession`s
  (and Drive services) wrap one shared `credentials` object. On a scan that runs
  past an OAuth token's ~1h lifetime, up to `fetch_workers` threads can each
  detect an expired token and refresh concurrently (no single-flight lock).
  Benign in the default service-account mode (refresh only rewrites token/expiry),
  but in OAuth mode a torn refresh can yield a 401, which `_request` does not
  retry, silently dropping that photo to the next resume pass. Cheap hardening:
  refresh once on the main thread before fanning out, and add 401 to the
  retry-after-refresh set in `DriveImageFetcher._request`.

- **No circuit-breaker on a total network outage.** With a sustained outage each
  photo burns ~7s of per-thread backoff (1+2+4s) before failing, so a long
  outage during a large scan means hours of no-progress sleeping before the run
  finishes (it stays correct and resumable — just slow). Add a shared
  consecutive-failure counter that short-circuits to fail-fast after K failures.

- **Per-thread sessions/services are never explicitly closed.** Bounded to
  `<= workers` per stage and GC-reclaimed when pool threads die, so negligible;
  add a `close()` only if fd pressure ever shows up.

## Scan speed (follow-ups)

- **similar stage is CLIP/MPS-bound, not fetch-bound.** On a first scan the near
  stage warms the cache, so `similar` mostly reads from disk and its wall-clock
  is the GPU embed pass. `embed_fetch_workers=32` barely helps there; the real
  lever is `clip_batch` (and MPS throughput), which must be tuned by measuring on
  the real machine. Decouple `embed_fetch_workers` from the fetch/hash default.

- **Each thumbnail is JPEG-decoded 2x on a first scan** (once for pHash/dHash in
  near, once for CLIP in similar; clustered members a 3rd time in
  `_is_color_variant`). Network is already paid once (disk cache hit). CPU-only,
  minor. Consider fusing hash+embed into one decode pass and stashing the HS
  histogram during hashing.

- **`_is_color_variant` fetches serially on the main thread.** Cache hits on a
  fresh scan, but cold serial round-trips on a resumed/incremental scan. Run it
  through `parallel_map`, or compute the histogram in the `fetch_and_hash` worker
  while the image is already decoded.

- **32-worker default vs. Drive CDN rate limits.** 32 is aggressive; if the
  `lh3.googleusercontent.com` thumbnail CDN starts returning 429s, backoff could
  make a scan slower. Can only be validated against a real library. If it shows
  up: add jitter to `_request` backoff, log when it backs off (a 429 burst
  currently looks like a hang), and consider lowering the default to
  `min(16, cpu*2)` with 32 as an opt-in ceiling. **Needs human tuning on the
  real photo library.**
