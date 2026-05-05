# M03 — Corpus providers + cache

**Goal:** corpora load from disk and from the web with on-disk caching
keyed by recipe-derived hashes. `pd-ocr-synth fetch gaelic` warms a
cache that `render` can read offline.

Spec: [`04-corpus-providers.md`](../specs/04-corpus-providers.md).

## Deliverables

### Provider interface

- [ ] `pd_ocr_synth.corpus.Provider` protocol matching spec 09:
      `fetch(ctx, options) -> Iterable[str]`,
      `cache_key(options) -> str`.
- [ ] `ProviderContext` (recipe_dir, cache_dir, http client, offline,
      logger).
- [ ] Registry of built-in providers, plus `python:` inline loader
      (deferred from M02).

### Built-in providers

Order of priority (Gaelic recipe needs the first three):

- [ ] `local` — file/glob/dir, parser inferred from extension.
- [ ] `web` — single URL with parser (`plain`, `html-text`, `tei-text`,
      `json`).
- [ ] `wikisource` — MediaWiki API; titles or category.
- [ ] `web_list` — many URLs.
- [ ] `hf_dataset` — `datasets.load_dataset` streaming.
- [ ] `internet_archive` — `_djvu.txt` derivative.
- [ ] `gutenberg` — Project Gutenberg ID with header/footer strip.

### HTTP infrastructure

- [ ] Shared `httpx.Client` with:
  - Polite default User-Agent (`pd-ocr-synth/<ver> (+contact)`)
  - Retries with exponential backoff
  - Per-host rate limit (1 req/s default)
  - Timeout (30s default, override per provider)
- [ ] `respect_robots: true` default; honor `robots.txt` for `web`/`web_list`.

### Cache layer

- [ ] Cache root: `${PD_OCR_SYNTH_CACHE:-~/.cache/pd-ocr-synth/}`.
- [ ] Each provider writes:
  - `<cache_root>/<provider>/<key>.txt` — parsed text
  - `<cache_root>/<provider>/<key>.meta.json` — URL, timestamp, sha256, size
- [ ] `--no-cache` bypass; `--offline` raises if cache miss.
- [ ] `pd-ocr-synth clean <recipe>` removes only that recipe's cache
      keys (matched via `cache_key`).

### Provider-level filters

- [ ] Apply `drop_lines_matching`, `keep_only_lines_matching`,
      `min_line_chars` post-fetch as documented in spec 04.

### CLI surface

- [ ] `pd-ocr-synth fetch <recipe>` — pre-fetch every provider with
      `cache: true`. Prints per-provider status: bytes fetched, cache
      hits, total time.
- [ ] `pd-ocr-synth describe gaelic` (extended from M02): now reports
      total chars across providers, unique tokens, top-10 character
      frequencies — useful for spotting under-covered codepoints
      before rendering.

### Tests

- [ ] Local provider against fixture files in `tests/fixtures/corpora/`.
- [ ] Web provider against `pytest-httpx` mock; verify caching, retry,
      timeout, robots.
- [ ] Wikisource provider against a recorded MediaWiki response.
- [ ] Cache invalidation when `cache_key` changes.
- [ ] Offline mode: missing cache → clean error message.

## Validation criteria

```bash
pd-ocr-synth fetch gaelic
# → fetches CELT pages and Wikisource pages; populates cache
ls ~/.cache/pd-ocr-synth/web/
# → shows .txt and .meta.json for each URL

pd-ocr-synth fetch gaelic    # second run
# → all hits served from cache, no network

pd-ocr-synth describe gaelic
# → "corpora: 4; total chars: 920341; unique tokens: 37412"
```

## Out of scope

- Tokenization into words/lines (that lives in M05 rendering, since it
  depends on the layout mode).
- Text transforms (M04).

## Risks / open items

- **MediaWiki API quirks.** Wikisource pages can be index pages or
  content; the provider needs to follow `<chapter>` links. Probably
  OK to start with title-only and improve in a follow-up.
- **`robots.txt` for CELT.** Verify in M03 that the polite defaults
  don't get blocked.
- **Cache size.** A few MB per recipe; not a concern for v1, but worth
  a `du -sh` check during testing.
