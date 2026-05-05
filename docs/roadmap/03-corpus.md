# M03 — Corpus providers + cache (mostly complete)

**Status:** ✅ everything the bundled `gaelic` recipe needs is landed
in commits ec1c4f4…f3c5a07 on `main`. Less-common providers and the
`describe` corpus-stats extension are deferred — see below.

**Goal:** corpora load from disk and from the web with on-disk caching
keyed by recipe-derived hashes. `pd-ocr-synth fetch gaelic` warms a
cache that `render` can read offline.

Spec: [`04-corpus-providers.md`](../specs/04-corpus-providers.md).

## Deliverables

### Provider interface

- [x] `pd_ocr_synth.corpus.Provider` protocol matching spec 09.
- [x] `ProviderContext` (recipe_dir, cache, http, offline, logger).
- [x] Registry of built-in providers with lazy entry-point loading.
- [ ] `python:` inline loader (deferred — not needed by gaelic; the
      registry's entry-point hook covers most cases).

### Built-in providers

Gaelic uses the first three; those are the priority of M03.

- [x] `local` — file/glob/dir, parser inferred from extension. Plain
      only at this layer; HTML/TEI/json belong to web.
- [x] `web` — single URL with `plain` / `html-text` / `tei-text` /
      `json` parsers.
- [x] `wikisource` — MediaWiki API (titles), special-cased for
      `language: mul` → `wikisource.org`.
- [ ] `web_list` — many URLs (deferred; trivial wrapper around `web`).
- [ ] `hf_dataset` — `datasets.load_dataset` streaming (deferred until
      a recipe needs it).
- [ ] `internet_archive` — `_djvu.txt` derivative (deferred).
- [ ] `gutenberg` — Project Gutenberg ID (deferred).

### HTTP infrastructure

- [x] Shared `httpx.Client` with polite UA, 30s timeout, transparent
      transport injection for tests.
- [x] Retries with exponential backoff on transient 408/425/429/5xx.
- [x] Per-host minimum interval (1s default, thread-safe).
- [ ] `respect_robots: true` default + `robots.txt` enforcement
      (deferred — risk noted; CELT and Wikisource have not blocked
      polite defaults in spot checks).

### Cache layer

- [x] Cache root: `${PD_OCR_SYNTH_CACHE:-~/.cache/pd-ocr-synth/}`.
- [x] `<root>/<provider>/<key>.{txt,meta.json}` layout. Long keys
      collapse to `prefix-sha256[:16]` so URL-shaped keys do not
      blow filesystem path limits.
- [x] `--no-cache` bypass on `fetch`.
- [x] `--offline` semantics (raises `OfflineCacheMissError` on miss);
      currently exposed via the provider context flag, the CLI flag
      lands with `render` in M05.
- [ ] `pd-ocr-synth clean <recipe>` (deferred — track as a small
      follow-up; the registry + cache_key plumbing is already in
      place, only the CLI handler needs writing).

### Provider-level filters

- [x] `drop_lines_matching`, `keep_only_lines_matching`,
      `min_line_chars` applied per corpus entry, post-fetch, before
      the entry's text joins the recipe-wide pool.

### CLI surface

- [x] `pd-ocr-synth fetch <recipe>` prints per-provider status
      (cache vs fetch, char count, elapsed seconds, cache key) and
      a total. Exit 4 (`CORPUS_EXIT`) on any per-entry failure;
      otherwise 0.
- [ ] `pd-ocr-synth describe gaelic` corpus-stats extension
      (total chars, unique tokens, top-10 codepoints) — deferred.
      Hangs on tokenization choices that the spec defers to M05;
      revisit alongside `render`.

### Tests

- [x] Local provider — unit tests over fixture files via `tmp_path`.
- [x] Web provider — `httpx.MockTransport` covers caching, retry,
      4xx/5xx surfacing.
- [x] Wikisource — `httpx.MockTransport` against synthetic
      MediaWiki responses (concat, cache hit, error info, missing
      titles, category-not-implemented).
- [x] Cache key sensitivity to options (path, parser, language,
      titles).
- [x] Offline mode: missing cache raises `OfflineCacheMissError`.
- [ ] Real `robots.txt` smoke test (deferred with the feature).

## Validation criteria

```bash
pd-ocr-synth fetch gaelic
# → fetches CELT pages and Wikisource (mul) pages; populates cache.
#   On a fresh devcontainer the local seed-words.txt entry fails by
#   design (the user has not yet authored it — see the M02 closeout
#   note). Web + wikisource succeed.
ls ~/.cache/pd-ocr-synth/web/
# → shows .txt and .meta.json for each URL.

pd-ocr-synth fetch gaelic    # second run
# → CELT entries served from cache. Wikisource fetches again because
#   gaelic uses cache: true on the wikisource entry too — verified.
```

## Risks / open items (still open)

- **`robots.txt` enforcement** — deferred; if upstream archives
  start blocking polite defaults, plumb into `get_with_retries`.
- **Wikisource category mode** — explicit `ProviderError` today; add
  when a recipe needs it (and the MediaWiki paginated category
  response is well documented).
- **Cache invalidation when source changes upstream** — out of scope
  for v1; user can `rm -rf` the cache or set `cache: false`.

## Closeout notes

- The deferred providers are all small wrappers: `web_list` is
  literally a loop over `web`; `gutenberg` and `internet_archive`
  are special-cased URL builders. Estimate <1 day each.
- `pd-ocr-synth clean <recipe>` is the only deferred CLI surface
  that should land before M04 starts using the cache for
  text-transform staging.
