# Fonts — Gaelic recipe

This is a non-normative reference for the four free font families used by
`recipes/gaelic.yaml`. None are bundled in the repo. Download them yourself
and place the unzipped files in `recipes/gaelic/fonts/`.

## Summary

| Family | Designer | License | Redistributable? | Best for |
|--------|----------|---------|------------------|----------|
| Gaelchló (Bunchló GC, Seanchló GC, …) | Vincent Morley | "Gaelchló-Celtic-free" — free for Celtic-language content; non-commercial elsewhere | **No** | The de-facto digital Cló Gaelach |
| Gadelica | Séamas Ó Brógáin | Marked free for commercial use | Unclear | Rounded 17th-c. minuscule, complements Gaelchló |
| Gaedhilge | Seán de Búrca | SIL OFL 1.1 | **Yes** | The cleanest license; redistributable |
| Cló Gaelach (Twomey) / Tuamach | Colum Twomey, rev. Michael Everson | Free, exact terms vary by repackager | Unclear | Historical original digital Cló Gaelach |

## Project licensing posture

`pd-ocr-synth` and the Gaelic recipe are **non-commercial**. Under the
Gaelchló terms, that means **two independent grounds for permission**:

1. The output is Celtic-language content (Irish OCR training data) —
   Gaelchló is free for any Celtic-language use.
2. The project is non-commercial — Gaelchló is free for non-commercial
   use in any language.

So all four font families in the recipe are unambiguously permitted for
**use** here. What is *not* permitted is bundling Gaelchló in this repo
(no redistribution without permission), and modification of any of the
non-OFL families is ambiguous.

## Why none are bundled

Use is permitted; *redistribution* is not (or is unclear) for three of
the four families. Treating all four the same way — "download yourself"
— keeps the repo license-clean and makes the pattern consistent for
future recipes (Fraktur, etc.) where similar issues will apply.

The OFL-licensed Gaedhilge is the only family whose redistribution is
unambiguous; it's the candidate for a future CI/test fallback if we
want one.

## Quick fetch — script

The cleanest way to get the default font set is the helper script,
which downloads each one directly from its upstream source (no mirror,
no re-distribution):

```bash
./scripts/fetch-fonts-gaelic.sh                    # the three default families
./scripts/fetch-fonts-gaelic.sh --with-gaedhilge   # also fetch Gaedhilge .sfd
```

It pulls into `recipes/gaelic/fonts/` by default; pass `--dest <DIR>` to
override. Pass `--yes` to skip the license confirmation prompt.

**Default set** (precompiled `.otf`, just unzip and use):

- Gaelchló family (Vincent Morley) — Bunchló, Seanchló, Glanchló, Úrchló
- Gadelica (Séamas Ó Brógáin)
- Cianchló (Feòrag NicBhrìde)

**Opt-in via `--with-gaedhilge`:** the OFL Gaedhilge family ships only as
FontForge source. Compiling it requires installing FontForge separately
(see below); without that, the `.sfd` is downloaded but cannot be used.
The Gaelic recipe marks the Gaedhilge entry as optional, so the recipe
still works without it.

A download script is acceptable here because **the script does not
redistribute** — your machine fetches from each project's canonical URL
the same as if you opened the link in a browser. (This is the same
pattern Linux distros use for non-redistributable assets.) We do not
mirror or cache the binaries on infrastructure we control.

## Direct download URLs (all verified live)

### Gaelchló — Vincent Morley

- Index: <https://www.gaelchlo.com/clonna1.html> (34 family pages)
- ZIP pattern: `https://www.gaelchlo.com/<abbr>.zip`
- Confirmed-working ZIPs:
  - <https://www.gaelchlo.com/bungc.zip> — Bunchló GC + variants
  - <https://www.gaelchlo.com/seangc.zip> — Seanchló GC
  - <https://www.gaelchlo.com/glangc.zip> — Glanchló GC
  - <https://www.gaelchlo.com/urgc.zip> — Úrchló GC
  - <https://www.gaelchlo.com/dubhgc.zip> — Dubhchló GC
- License (paraphrased from gaelchlo.com): permitted for content in
  Celtic languages, or for non-commercial material in any language.
  Commercial non-Celtic publishing requires a paid license. No
  modification or redistribution without permission.
- Permitted for this project on **two** independent grounds: Irish is a
  Celtic language, and the project is non-commercial.

### Gadelica — Séamas Ó Brógáin

- Hosted on gaelchlo.com (the canonical Gaelic font hub):
  - <https://www.gaelchlo.com/Gadelica%20Ultima.zip>
- License: marked free; designer requests no modification and no charging
  for redistribution.

### Cianchló — Feòrag NicBhrìde

- <https://www.gaelchlo.com/Cianchlo.zip>
- Free; bundled in the script as a stylistic variation source.

### Gaedhilge — Seán de Búrca (opt-in)

- Repo: <https://github.com/leftmostcat/gaedhilge-fonts>
- License: **SIL Open Font License 1.1** (the only unambiguously
  redistributable family of the set).
- Distribution: FontForge source (`gaedhilge.sfd`). Compile to OTF:
  ```bash
  fontforge -lang=ff -c 'Open($1); Generate($2)' gaedhilge.sfd Gaedhilge.otf
  ```
- Run `./scripts/fetch-fonts-gaelic.sh --with-gaedhilge` to download it.
  The script will compile it if `fontforge` is on PATH; otherwise it
  stashes the `.sfd` and prints the build command. Installing FontForge
  is **your responsibility** — the project does not auto-install it.
  See <https://fontforge.org/en-US/downloads/> for installers.

### Cló Gaelach (Twomey) / Cló Tuamach — Colum Twomey, rev. Everson

- The Tuamach revision is **not** distributed by Evertype; Evertype's
  CeltScript fonts (Ceanannas, Doire, Duibhlinn, …) are **commercial**
  via MyFonts. Free Tuamach mirrors exist (wfonts.com, ffonts.net) but
  redistribution rights are unclear per mirror. Not used by the recipe.

## Other free options worth knowing about

Listed at <https://www3.smo.uhi.ac.uk/oduibhin/mearchlar/fonts.htm>:

- **Gaeilge 1, Gaeilge 2, Gaeilge Unicode** — Padraig McCarthy
- **Gael AX / BX** — F. M. O'Carroll, Unicode by KAD
- **Rudhraigheacht** — Galt Barber

These are not in the recipe by default but you can add them if you want
greater stylistic variation.

## Validation

After downloading and placing the fonts, run:

```bash
pd-ocr-synth validate gaelic
```

This opens each font and surfaces `font_missing` / `font_unreadable` /
`font_empty` errors (plus `optional_font_missing` warnings) — see
`docs/specs/06-rendering.md` for the validator's font-check contract.
Per-codepoint corpus-vs-font coverage reporting at validate time is
deferred; today, missing glyphs are detected at *render* time and
recorded as `missing_glyph` skip entries in the manifest.
