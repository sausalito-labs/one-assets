#!/usr/bin/env bash
# Headless preview: extract geometry in Blender, rasterise with the browser sim.
#   scripts/preview.sh [view ...]     views: front 3q side back head face legs
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLEND="${BLEND:-$ROOT/characters/takeru/takeru.blend}"
OUT="${OUT:-/tmp/opencode}"
SUBSURF="${SUBSURF:-1}"
SIZE="${SIZE:-640}"
mkdir -p "$OUT"
blender --background --python "$ROOT/scripts/extract_mesh.py" -- \
  --blend "$BLEND" --npz "$OUT/mesh.npz" --subsurf "$SUBSURF" --with-ink 2>&1 \
  | grep -E "^\[extract\]" || true
views=("$@"); [ ${#views[@]} -eq 0 ] && views=(front 3q face)
for v in "${views[@]}"; do
  python3 "$ROOT/scripts/rasterize.py" --npz "$OUT/mesh.npz" \
    --out "$OUT/$v.png" --view "$v" --size "$SIZE" --browser
done
