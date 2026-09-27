"""Build a T-pose Muay Thai fighter with separated limbs + armature.

Reference: lean ONE Championship weigh-in photo (flexing, bleached hair,
red trunks). Model is re-posed to a clean T-pose and given red Muay Thai
shorts. Stylized likeness: proportions/hair/shorts matched, not a scan.

Usage (inside `enter-workflow.sh 3d`):
    blender --background --python scripts/build_human_tpose.py -- \\
        --out characters/human_tpose/human_tpose.blend \\
        --fbx exports/human_tpose.fbx \\
        --glb demo/fighter/fighter.glb

Conventions: character faces +Y, up is +Z, arms extend along +/-X (T-pose).
Every body part is its own object, rigid-bound (one vertex group each) to
its armature bone so limbs stay separated but pose together.
"""
import argparse
import math
import sys

try:
    import bpy
    from mathutils import Vector
except ImportError:  # allow --help outside Blender
    bpy = None


# ---------------------------------------------------------------- materials
def mat(name, color, roughness=0.6, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    return m


# ---------------------------------------------------------------- mesh helpers
def _finish(obj, material, bone, smooth=False):
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    if material is not None:
        obj.data.materials.append(material)
    # rigid bind: every vert fully weighted to one bone
    arm = bpy.data.objects["FighterArmature"]
    mod = obj.modifiers.new("ArmatureBind", "ARMATURE")
    mod.object = arm
    vg = obj.vertex_groups.new(name=bone)
    vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    return obj


def cube(name, loc, scale, material, bone):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc, scale=scale)
    return _finish(bpy.context.active_object, material, bone)


def ball(name, loc, radius, material, bone, scale=(1, 1, 1), smooth=True):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=loc, scale=scale,
        segments=24, ring_count=16)
    obj = bpy.context.active_object
    obj.name = name
    return _finish(obj, material, bone, smooth=smooth)


def cyl(name, loc, radius, depth, material, bone, rotation=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=loc, rotation=rotation,
        vertices=20)
    obj = bpy.context.active_object
    obj.name = name
    return _finish(obj, material, bone, smooth=True)


def torus(name, loc, major, minor, material, bone):
    bpy.ops.mesh.primitive_torus_add(
        location=loc, major_radius=major, minor_radius=minor,
        major_segments=24, minor_segments=12)
    obj = bpy.context.active_object
    obj.name = name
    return _finish(obj, material, bone, smooth=True)


# ---------------------------------------------------------------- armature
BONES = {
    # name: (head, tail, parent)
    "root": ((0, 0, 0), (0, 0, 0.25), None),
    "spine": ((0, 0, 1.00), (0, 0, 1.30), "root"),
    "chest": ((0, 0, 1.30), (0, 0, 1.50), "spine"),
    "neck": ((0, 0, 1.50), (0, 0, 1.60), "chest"),
    "head": ((0, 0, 1.60), (0, 0, 1.84), "neck"),
    "upper_arm.L": ((0.19, 0, 1.47), (0.47, 0, 1.47), "chest"),
    "forearm.L": ((0.47, 0, 1.47), (0.72, 0, 1.47), "upper_arm.L"),
    "hand.L": ((0.72, 0, 1.47), (0.88, 0, 1.47), "forearm.L"),
    "upper_arm.R": ((-0.19, 0, 1.47), (-0.47, 0, 1.47), "chest"),
    "forearm.R": ((-0.47, 0, 1.47), (-0.72, 0, 1.47), "upper_arm.R"),
    "hand.R": ((-0.72, 0, 1.47), (-0.88, 0, 1.47), "forearm.R"),
    "thigh.L": ((0.11, 0, 1.00), (0.11, 0, 0.55), "spine"),
    "shin.L": ((0.11, 0, 0.55), (0.11, 0, 0.12), "thigh.L"),
    "foot.L": ((0.11, 0, 0.12), (0.11, 0.24, 0.05), "shin.L"),
    "thigh.R": ((-0.11, 0, 1.00), (-0.11, 0, 0.55), "spine"),
    "shin.R": ((-0.11, 0, 0.55), (-0.11, 0, 0.12), "thigh.R"),
    "foot.R": ((-0.11, 0, 0.12), (-0.11, 0.24, 0.05), "shin.R"),
}


def build_armature():
    arm_data = bpy.data.armatures.new("FighterArmature")
    arm = bpy.data.objects.new("FighterArmature", arm_data)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    edits = {}
    for name, (head, tail, _parent) in BONES.items():
        b = arm_data.edit_bones.new(name)
        b.head = Vector(head)
        b.tail = Vector(tail)
        edits[name] = b
    for name, (_h, _t, parent) in BONES.items():
        if parent:
            edits[name].parent = edits[parent]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


# ---------------------------------------------------------------- body
def build_body(S, H, RED, WHITE, GOLD, DARK):
    X90 = (0, math.radians(90), 0)  # cylinder Z-axis -> X-axis (arms)
    parts = []
    # torso column (lean, defined midsection like the photo)
    parts.append(cube("Pelvis", (0, 0, 1.02), (0.150, 0.100, 0.090), S, "spine"))
    parts.append(cube("Torso", (0, 0, 1.24), (0.165, 0.105, 0.130), S, "spine"))
    parts.append(cube("Chest", (0, 0, 1.43), (0.190, 0.115, 0.090), S, "chest"))
    parts.append(cyl("Neck", (0, 0, 1.565), 0.060, 0.09, S, "neck"))
    # head + face (stylized)
    parts.append(ball("Head", (0, 0, 1.705), 0.115, S, "head"))
    parts.append(ball("Eye.L", (0.045, 0.102, 1.725), 0.016, DARK, "head"))
    parts.append(ball("Eye.R", (-0.045, 0.102, 1.725), 0.016, DARK, "head"))
    parts.append(cube("Mouth", (0, 0.101, 1.655), (0.030, 0.006, 0.008),
                      DARK, "head"))
    # bleached textured crop: cap + fringe
    parts.append(ball("Hair", (0, -0.012, 1.770), 0.120, H, "head",
                      scale=(1.02, 1.02, 0.62)))
    parts.append(cube("HairFringe", (0, 0.095, 1.745), (0.085, 0.025, 0.035),
                      H, "head"))
    # gold chain from the photo
    parts.append(torus("Chain", (0, 0, 1.560), 0.068, 0.008, GOLD, "chest"))
    # arms, T-pose along X
    for side, s in (("L", 1), ("R", -1)):
        parts.append(cyl(f"UpperArm.{side}", (s * 0.330, 0, 1.47),
                         0.058, 0.30, S, f"upper_arm.{side}", X90))
        parts.append(cyl(f"Forearm.{side}", (s * 0.595, 0, 1.47),
                         0.048, 0.28, S, f"forearm.{side}", X90))
        parts.append(ball(f"Hand.{side}", (s * 0.815, 0, 1.47), 0.075, S,
                          f"hand.{side}", scale=(1.25, 0.70, 0.95)))
    # legs
    for side, s in (("L", 1), ("R", -1)):
        parts.append(cyl(f"Thigh.{side}", (s * 0.11, 0, 0.760),
                         0.088, 0.46, S, f"thigh.{side}"))
        parts.append(cyl(f"Shin.{side}", (s * 0.11, 0, 0.315),
                         0.058, 0.44, S, f"shin.{side}"))
        parts.append(cube(f"Foot.{side}", (s * 0.11, 0.065, 0.040),
                          (0.050, 0.120, 0.038), S, f"foot.{side}"))
    # ---- red Muay Thai shorts, high-cut wide legs ----
    parts.append(cube("Shorts", (0, 0, 0.985), (0.190, 0.140, 0.125), RED, "spine"))
    for side, s in (("L", 1), ("R", -1)):
        parts.append(cyl(f"ShortLeg.{side}", (s * 0.11, 0, 0.825),
                         0.108, 0.20, RED, f"thigh.{side}"))
        parts.append(cyl(f"LegTrim.{side}", (s * 0.11, 0, 0.730),
                         0.109, 0.025, WHITE, f"thigh.{side}"))
    parts.append(cube("Waistband", (0, 0, 1.098), (0.192, 0.142, 0.028),
                      WHITE, "spine"))
    # gold side stripes
    parts.append(cube("Stripe.L", (0.192, 0, 0.985), (0.006, 0.100, 0.110),
                      GOLD, "spine"))
    parts.append(cube("Stripe.R", (-0.192, 0, 0.985), (0.006, 0.100, 0.110),
                      GOLD, "spine"))
    return parts


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    S = mat("Skin", (0.87, 0.66, 0.53), roughness=0.55)
    H = mat("HairBleach", (0.82, 0.79, 0.72), roughness=0.85)
    RED = mat("ShortsRed", (0.72, 0.05, 0.09), roughness=0.7)
    WHITE = mat("TrimWhite", (0.93, 0.93, 0.94), roughness=0.6)
    GOLD = mat("Gold", (0.83, 0.62, 0.25), roughness=0.35, metallic=0.8)
    DARK = mat("Dark", (0.08, 0.06, 0.05), roughness=0.5)
    build_armature()  # must exist before _finish binds modifiers
    parts = build_body(S, H, RED, WHITE, GOLD, DARK)
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        o.select_set(True)
    bpy.data.objects["FighterArmature"].select_set(True)
    print(f"[build] fighter done: {len(parts)} parts + armature, "
          f"{len(BONES)} bones")


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--fbx", default=None)
    p.add_argument("--glb", default=None)
    return p.parse_args(argv)


def main(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parse_args(argv)
    if bpy is None:
        raise RuntimeError("Run inside Blender: blender --background --python "
                           "scripts/build_human_tpose.py")
    build()
    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[build] saved {args.out}")
    if args.fbx:
        bpy.ops.export_scene.fbx(filepath=args.fbx, use_selection=False,
                                 add_leaf_bones=False)
        print(f"[build] exported {args.fbx}")
    if args.glb:
        bpy.ops.export_scene.gltf(filepath=args.glb, export_format="GLB",
                                  use_selection=True)
        print(f"[build] exported {args.glb}")


if __name__ == "__main__":
    main(sys.argv)
