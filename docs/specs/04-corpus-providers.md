# 04 — Corpus providers

A `corpus` entry has a `type` that selects a provider. All providers emit
a stream of UTF-8 text to a shared post-processing pipeline (transforms +
tokenization).

## Common keys

These apply across all providers:

| Key | Default | Meaning |
|-----|---------|---------|
| `cache` | `true` | Cache the fetched/parsed text on disk |
| `cache_key` | derived | Override the cache key (advanced) |
| `max_chars` | unlimited | Truncate after this many UTF-8 chars |
| `min_word_length` | `1` | Drop tokens shorter than this after tokenization |
| `language` | unset | Informational; may guide tokenization |

The cache lives at `${PD_OCR_SYNTH_CACHE:-~/.cache/pd-ocr-synth/}`.

## `local`

Read text from local files.

```yaml
- type: local
  path: ./corpora/seed-words.txt
```

`path` may be a single file, a glob (`./corpora/*.txt`), or a directory
(read recursively, alphabetical order).

Parsers are inferred from extension (`.txt` plain, `.html` html-text,
`.xml` tei-text) or set explicitly with `parser:`.

## `web`

HTTP GET a single URL.

```yaml
- type: web
  url: https://celt.ucc.ie/published/G100001A.html
  parser: html-text          # plain | html-text | tei-text | json
  cache: true
  user_agent: "pd-ocr-synth/0.1 (+contact@example.com)"
  retries: 3
  timeout_seconds: 30
```

Parsers:

| Parser | Behavior |
|--------|----------|
| `plain` | Use response body as-is |
| `html-text` | BeautifulSoup → drop `<script>/<style>` → text |
| `tei-text` | TEI XML → extract `<text>` content (CELT, OpenGreekAndLatin) |
| `json` | Apply `field_path` (e.g. `$.entries[*].body`) to extract strings |

Web fetches honor `robots.txt` if `respect_robots: true` (default
`true`). Polite defaults: 1 req/sec per host, 30s timeout, 3 retries
with backoff.

## `web_list`

Fetch many URLs at once.

```yaml
- type: web_list
  urls:
    - https://example.com/page-1.html
    - https://example.com/page-2.html
  parser: html-text
  cache: true
```

Or load URLs from a file:

```yaml
- type: web_list
  urls_file: ./corpora/urls.txt
  parser: html-text
```

## `wikisource`

Pull pages from a Wikisource language edition.

```yaml
- type: wikisource
  language: ga                 # ISO language code; ga = Irish
  titles:
    - "Séadna"
    - "Mo Sgéal Féin"
  cache: true
```

Or by category:

```yaml
- type: wikisource
  language: ga
  category: "Téacsleabhair sa Ghaeilge"
  max_pages: 50
```

Uses the MediaWiki API; output is plain text with wiki markup stripped.

## `hf_dataset`

Hugging Face Datasets streaming.

```yaml
- type: hf_dataset
  name: example/irish-corpus
  split: train
  field: text                  # column to extract
  max_rows: 10000
  cache: true
```

Honors `HF_HOME` for the underlying cache; recipe-level cache stores the
extracted text per the common cache rules.

## `internet_archive`

Pull plain-text from Internet Archive items (using the `_djvu.txt` derivative
when available).

```yaml
- type: internet_archive
  identifiers:
    - oireachtas1900
    - irish-grammar-1908
  cache: true
```

## `gutenberg`

Project Gutenberg by ID.

```yaml
- type: gutenberg
  ids: [12345, 23456]
  cache: true
```

Strips the standard Gutenberg header/footer.

## Caching

The first time a recipe runs, every provider with `cache: true` writes its
parsed output to `${PD_OCR_SYNTH_CACHE}/<provider>/<key>.txt` plus a
sidecar `<key>.meta.json` (URL, fetched-at, byte size, sha256).

Subsequent runs read from cache without network access. `pd-ocr-synth fetch
<recipe>` warms the cache up front; `pd-ocr-synth render` and `preview`
will refuse to hit the network if `--offline` is set.

`pd-ocr-synth clean <recipe>` removes only the cache entries owned by that
recipe (matched by `cache_key`).

## Tokenization

After all providers concatenate and transforms apply, the result is split
according to the chosen `layout.mode`:

| Layout mode | Tokenization |
|-------------|--------------|
| `word_crops` | Whitespace + punctuation split → one word per sample |
| `lines` | Paragraph break or `\n` → one line per sample |
| `paragraphs` | Blank line → one paragraph per sample |
| `pages` | Multiple paragraphs flowed into a page region |

Sampling is with replacement, weighted toward unique tokens to avoid
overfitting on stop-words. Override the weighting with
`corpus_sampling: uniform | unique_weighted | frequency`.

## Provider-level filters

Each provider may define a post-fetch filter:

```yaml
- type: web
  url: https://example.com/dictionary.html
  parser: html-text
  filter:
    drop_lines_matching: '^[0-9]+$'   # regex
    keep_only_lines_matching: '\b[A-Za-zÁÉÍÓÚáéíóúḃċḋḟġṁṗṡṫ]+\b'
    min_line_chars: 2
```

These are independent of the recipe-level `text_transforms`; provider
filters are about cleaning *source* data before it joins the pool.

## Authoring a custom provider

See [09 — Extending](09-extending.md). A provider is a Python class
implementing `fetch(recipe_dir, options) -> Iterable[str]`, registered via
an entry point.
