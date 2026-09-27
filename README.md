# one-assets

3D assets for one-arcade — Blender characters, props, environments.

Workflow: `enter-workflow.sh 3d` (Blender + Godot + Python + ImageMagick)

## Layout

- `characters/` — per-character Blender sources, T-pose + rigged
  - `characters/human_tpose/` — first human from photo (WIP, waiting on reference)
- `props/` — small props
- `environments/` — sets / levels
- `textures/` — shared textures / materials
- `scripts/` — Blender Python build scripts (headless: `blender --background --python`)
- `exports/` — exported `.fbx` / `.glb` (generated, but committed for now)
- `reference/` — source photos (DO NOT commit private photos — local only, gitignored)

## Human T-pose convention

Each human:
- T-pose, facing +Y, up +Z (Blender default)
- Separate objects: `Torso`, `Head`, `UpperArm.L/R`, `Forearm.L/R`, `Hand.L/R`, `Thigh.L/R`, `Shin.L/R`, `Foot.L/R`, `Pelvis`
- One `Armature` with matching bones, automatic weights, `.blend` + `.fbx` in `exports/`

## Build

```bash
enter-workflow.sh 3d
blender --background --python scripts/build_human_tpose.py -- --out characters/human_tpose/human_tpose.blend
# export FBX from Blender or via script flag --fbx
```
