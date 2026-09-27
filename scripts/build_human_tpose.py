"""Build a T-pose human with separated limbs + armature.

Usage (inside `enter-workflow.sh 3d`):
    blender --background --python scripts/build_human_tpose.py -- --out characters/human_tpose/human_tpose.blend --fbx exports/human_tpose.fbx

Current status: placeholder proportions. Once the reference photo lands,
tune HEIGHT, limb lengths, colors, head/hair to match.
"""
import argparse
import sys

# NOTE: this script runs inside Blender (bpy). Import guarded so `python3
# scripts/build_human_tpose.py --help` still works outside Blender.
try:
    import bpy
except ImportError:
    bpy = None


HEIGHT = 1.75  # m, tune from photo


def build():
    if bpy is None:
        raise RuntimeError("Run this script with: blender --background --python scripts/build_human_tpose.py")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # TODO: build separated limb meshes (cubes -> shaped), T-pose transforms,
    # armature, parent with automatic weights, materials from photo.
    # Placeholder cube so pipeline (blend + fbx export) is testable end-to-end.
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    body = bpy.context.active_object
    body.name = "Torso"
    print(f"[build_human_tpose] placeholder Torso created, target height={HEIGHT}m")


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="output .blend path")
    p.add_argument("--fbx", default=None, help="optional output .fbx path")
    return p.parse_args(argv)


def main(argv):
    # Blender swallows args after `--`, so accept raw argv past `--`.
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parse_args(argv)
    build()
    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[build_human_tpose] saved {args.out}")
    if args.fbx:
        bpy.ops.export_scene.fbx(filepath=args.fbx, use_selection=False)
        print(f"[build_human_tpose] exported {args.fbx}")


if __name__ == "__main__":
    main(sys.argv)
