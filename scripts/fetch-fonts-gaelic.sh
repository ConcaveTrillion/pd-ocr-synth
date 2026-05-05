#!/usr/bin/env bash
# fetch-fonts-gaelic.sh — download free Gaelic fonts from upstream sources.
#
# This script does NOT redistribute fonts. It downloads from each project's
# canonical URL on your behalf, the same as if you opened the link yourself.
# Files land in <dest>/ ready for `recipes/gaelic.yaml`.
#
# Usage:
#   ./scripts/fetch-fonts-gaelic.sh [--dest DIR] [--yes] [--with-gaedhilge]
#
# Defaults destination to: recipes/gaelic/fonts/
#
# By default, fetches the six precompiled OTF families from gaelchlo.com.
# Gaedhilge (the OFL family) ships only as FontForge source (.sfd) and
# requires FontForge installed separately to compile to .otf, so it is
# opt-in via --with-gaedhilge. The Gaelic recipe lists it as optional —
# the recipe still works without it.
#
# Sources used (each fetched directly from its origin, no mirroring):
#   - https://www.gaelchlo.com/  (Vincent Morley + bundled free fonts)
#   - https://github.com/leftmostcat/gaedhilge-fonts  (with --with-gaedhilge)

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/recipes/gaelic/fonts"
ASSUME_YES=0
WITH_GAEDHILGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --with-gaedhilge) WITH_GAEDHILGE=1; shift ;;
    -h|--help)
      sed -n '2,21p' "$0" | sed 's/^# \?//'
      exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cat <<'NOTICE'
====================================================================
Free Gaelic fonts — license notice (please read)
====================================================================
You are about to download three free font families directly from their
upstream sources. This script does not modify or redistribute them; it
just automates the download.

  1. Gaelchló (Vincent Morley) — https://www.gaelchlo.com/
       Free for Celtic-language content OR non-commercial use in any
       language. No modification or redistribution without permission.
  2. Gadelica (Séamas Ó Brógáin) — distributed via gaelchlo.com.
       Marked free; designer asks: do not modify, do not charge for
       redistribution.
  3. Cianchló (Feòrag NicBhrìde) — distributed via gaelchlo.com. Free.

Optional (--with-gaedhilge):
  4. Gaedhilge (Seán de Búrca) — https://github.com/leftmostcat/gaedhilge-fonts
       SIL Open Font License 1.1. Ships as FontForge source (.sfd);
       you must install FontForge separately to compile it.

By continuing you confirm your use complies with each license.
====================================================================
NOTICE

if [[ "$ASSUME_YES" -ne 1 ]]; then
  read -r -p "Proceed? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "aborted."; exit 1; }
fi

mkdir -p "$DEST"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fetch_zip() {
  local label="$1" url="$2" zipname
  zipname="$(basename "${url%%\?*}")"
  echo "→ $label"
  echo "  $url"
  curl -fsSL --retry 3 -o "$TMP/$zipname" "$url"
  # Some ZIPs use CP1252-encoded filenames; try the convert flag, fall
  # back to plain extraction.
  unzip -O CP1252 -oq "$TMP/$zipname" -d "$DEST" 2>/dev/null \
    || unzip -oq "$TMP/$zipname" -d "$DEST"
  echo "  ✓ unpacked to $DEST"
}

# Some ZIPs use Latin-1 (CP1252) filenames; on a UTF-8 filesystem those
# bytes are not valid UTF-8 and tools mis-render them. Re-encode each
# extracted directory name from Latin-1 to ASCII (transliterating any
# diacritics) so YAML paths stay simple.
normalize_dirs() {
  local d clean orig
  for d in "$DEST"/*; do
    [[ -d "$d" ]] || continue
    orig="$(basename "$d")"
    # Detect any non-ASCII byte (>= 0x80) without relying on grep's
    # collation, which varies by libc.
    if LC_ALL=C awk 'BEGIN{exit !match(ARGV[1], /[\200-\377]/)}' "$orig"; then
      clean="$(printf %s "$orig" | iconv -f LATIN1 -t ASCII//TRANSLIT 2>/dev/null || echo "$orig")"
      if [[ -n "$clean" && "$clean" != "$orig" ]]; then
        mv "$d" "$DEST/$clean"
        echo "  ↳ normalized: $orig → $clean"
      fi
    fi
  done
}

# Vincent Morley's Gaelchló — pick a few representative families. Add more by
# pattern: https://www.gaelchlo.com/<abbr>.zip (see clonna1.html for the
# full list).
fetch_zip "Bunchló GC"  "https://www.gaelchlo.com/bungc.zip"
fetch_zip "Seanchló GC" "https://www.gaelchlo.com/seangc.zip"
fetch_zip "Glanchló GC" "https://www.gaelchlo.com/glangc.zip"
fetch_zip "Úrchló GC"   "https://www.gaelchlo.com/urgc.zip"

# Other free fonts hosted on gaelchlo.com (note the URL-encoded space).
fetch_zip "Gadelica Ultima" "https://www.gaelchlo.com/Gadelica%20Ultima.zip"
fetch_zip "Cianchló"        "https://www.gaelchlo.com/Cianchlo.zip"

# --- Gaedhilge (opt-in via --with-gaedhilge) ---------------------------------
# OFL FontForge source. Requires FontForge installed separately to compile.
# The recipe marks it optional, so skipping is fine.
if [[ "$WITH_GAEDHILGE" -eq 1 ]]; then
  echo "→ Gaedhilge (OFL source) [--with-gaedhilge]"
  SFD_URL="https://raw.githubusercontent.com/leftmostcat/gaedhilge-fonts/master/gaedhilge.sfd"
  echo "  $SFD_URL"
  curl -fsSL --retry 3 -o "$DEST/gaedhilge.sfd" "$SFD_URL"
  if command -v fontforge >/dev/null 2>&1; then
    fontforge -lang=ff -c 'Open($1); Generate($2)' "$DEST/gaedhilge.sfd" "$DEST/Gaedhilge.otf" \
      >/dev/null 2>&1 \
      && echo "  ✓ built $DEST/Gaedhilge.otf" \
      || echo "  ! FontForge build failed; the .sfd remains for manual build"
  else
    echo "  ! fontforge not on PATH. Install FontForge separately, then run:"
    echo "      fontforge -lang=ff -c 'Open(\$1); Generate(\$2)' $DEST/gaedhilge.sfd $DEST/Gaedhilge.otf"
  fi
fi

normalize_dirs

echo
echo "Done. Files in $DEST:"
find "$DEST" -type f \( -iname '*.otf' -o -iname '*.ttf' -o -iname '*.sfd' \) \
  | sort | sed "s|^$DEST/|  |"
