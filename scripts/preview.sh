#!/usr/bin/env bash
# Headless preview: extract geometry in Blender, rasterize with PIL.
# Run inside the `3d` workflow shell:
#   scripts/preview.sh [view ...]
# Views default to a full turnaround set. Set SUBSURF=0 for speed.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BLEND="${BLEND:-$ROOT/characters/human_tpose/human_tpose.blend}"
OUTDIR="${OUTDIR:-/tmp/opencode}"
SUBSURF="${SUBSURF:-1}"
SIZE="${SIZE:-560}"
NPZ="$OUTDIR/fighter.npz"

mkdir -p "$OUTDIR"
blender --background --python "$ROOT/scripts/softrender.py" -- \
    --blend "$BLEND" --npz "$NPZ" --subsurf "$SUBSURF" 2>&1 \
    | grep -E "^\[extract\]" || true

views=("$@")
if [ ${#views[@]} -eq 0 ]; then
    views=(front threeq side head)
fi
for v in "${views[@]}"; do
    python3 "$ROOT/scripts/rasterize.py" --npz "$NPZ" \
        --out "$OUTDIR/sr_$v.png" --view "$v" --size "$SIZE"
done
