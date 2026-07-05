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

## Review-page query optimizations (from the Phases 0–6 code review)

- **`review_all` / `review_batch` / `set_group_reviewed` are N+1.** They loop the
  filtered groups and call `_group_context` (3–4 queries) + `_finalize_group`
  (one upsert per member) per group, holding the sqlite writer for the whole
  loop. Sub-second on a personal library but linear in group count. Rewrite the
  default-fill as one set-based `INSERT ... SELECT` guarded by `NOT EXISTS` (to
  preserve manual decisions), and/or chunk the commit.

- **`_review_page_ids` total-count does needless work.** The `COUNT(*)` wraps the
  full base query (a `JOIN photos` + `MIN/MAX(score)` aggregates) that a row
  count doesn't need, and there's no index on `groups(tier)` so the outer query
  scans groups. Compute the count from a lean subquery (no photos join, no score
  aggregates) and add `CREATE INDEX idx_groups_tier ON groups(tier)`. Pre-existing
  (not introduced by the phase work); read-only and sub-second today.

- **Optional `idx_photos_folder_path`.** The brand-folder palette query was fixed
  (MIN(id) instead of a correlated subquery); an index on `photos(folder_path)`
  would speed its `GROUP BY` further if the library grows very large.

## Move-to-Trash (follow-ups from the write-OAuth work)

- **`load_trash_oauth_credentials()` is called twice per confirm-page load** —
  once via `can_trash()` and once via `trash_owner_connected()`. Each reads the
  token file (and may refresh it, though refresh is gated on a local expiry
  check, so no network unless the token is actually expired). Trivial; memoize
  per-request or compute the owner-connected flag once and thread it through.

- **Preflight `capabilities.canTrash` before the trash loop.** Instead of firing
  a doomed `files.update` per non-owned file and classifying the 403, fetch
  `capabilities/canTrash,ownedByMe` up front (store during sync or a cheap
  `files.get`) so the confirm page can warn precisely and skip doomed calls.

## WebP → PNG conversion (follow-ups)

- **Partial-failure leaves the WebP un-relocated.** If `upload_file` succeeds
  but `move_file` then fails (transient), the pass records nothing and retries
  next scan — but the retry sees the now-existing target PNG and skips with "a
  PNG of that name exists", so the original WebP never moves to the trash folder
  (it just sits beside the new PNG). Safe (no data loss, no duplicate PNG), but
  the original isn't tidied. Fix: when the existing target PNG's id matches a
  `webp_conversions.png_drive_id` we created, complete the move instead of
  skipping; otherwise keep skipping (never assume an unrelated PNG is ours).

- **No UI control or results view.** Conversion is config-only (`[webp]
  convert_after_scan`) and reports via logs + the `webp_conversions` table. A
  settings toggle, a manual "Convert WebPs now" button, and a small results
  summary (converted / skipped / space reclaimed) would make it discoverable.

- **Serial conversion.** Downloads + uploads run one-at-a-time in the post-scan
  thread. Fine for modest WebP counts; parallelize via `parallel_map` if a
  library has thousands of WebPs.
