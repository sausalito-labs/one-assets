"""Build an SFIV-style (Xbox 360 era) T-pose Muay Thai fighter.

Reference: lean ONE Championship weigh-in photo (bleached hair, gold
chain, red trunks) -> re-posed to a clean T-pose, red Muay Thai shorts,
red hand/ankle wraps. Heroic Street Fighter proportions: wide shoulders,
big fists, sculpted muscle masses.

Method: smooth subdivided base meshes shaped programmatically
(tapers, bulges, jawline) instead of raw primitives. Every part stays
its own object, rigid-bound (one vertex group each) to its armature
bone, so limbs stay separated but pose together.

Usage (inside `enter-workflow.sh 3d`):
    blender --background --python scripts/build_human_tpose.py -- \\
        --out characters/human_tpose/human_tpose.blend \\
        --fbx exports/human_tpose.fbx \\
        --glb demo/fighter/fighter.glb

Conventions: faces +Y, up +Z, arms along +/-X. Bone names are the
one-arcade slicer contract (render_fighter_parts.py keywords) - do NOT
rename bones without updating the slicer.
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
def mat(name, color, roughness=0.6, metallic=0.0, subsurface=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if "Metallic" in bsdf.inputs:
        bsdf.inputs["Metallic"].default_value = metallic
    if subsurface:
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = subsurface
        elif "Subsurface" in bsdf.inputs:  # older Principled
            bsdf.inputs["Subsurface"].default_value = subsurface
    return m


# ---------------------------------------------------------------- mesh finish
def finish(obj, material, bone, subsurf=0, smooth=True):
    if smooth:
        for poly in obj.data.polygons:
            poly.use_smooth = True
    if material is not None:
        obj.data.materials.append(material)
    arm = bpy.data.objects["FighterArmature"]
    mod = obj.modifiers.new("ArmatureBind", "ARMATURE")
    mod.object = arm
    if subsurf:
        sm = obj.modifiers.new("Smooth", "SUBSURF")
        sm.levels = subsurf
        sm.render_levels = subsurf
    vg = obj.vertex_groups.new(name=bone)
    vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    return obj


def cube(name, loc, scale, material, bone, subsurf=0):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc, scale=scale)
    obj = bpy.context.active_object
    obj.name = name
    return finish(obj, material, bone, subsurf=subsurf)


def ball(name, loc, radius, material, bone, scale=(1, 1, 1), subsurf=0,
         seg=24, rings=16):
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius, location=loc, scale=scale,
        segments=seg, ring_count=rings)
    obj = bpy.context.active_object
    obj.name = name
    return finish(obj, material, bone, subsurf=subsurf)


def cyl(name, loc, radius, depth, material, bone, rotation=(0, 0, 0),
        subsurf=0, verts=24):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=loc, rotation=rotation,
        vertices=verts)
    obj = bpy.context.active_object
    obj.name = name
    return finish(obj, material, bone, subsurf=subsurf)


def torus(name, loc, major, minor, material, bone):
    bpy.ops.mesh.primitive_torus_add(
        location=loc, major_radius=major, minor_radius=minor,
        major_segments=24, minor_segments=12)
    obj = bpy.context.active_object
    obj.name = name
    return finish(obj, material, bone)


# ---------------------------------------------------------------- shaping
def _coords(obj):
    import numpy as np
    me = obj.data
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    return me, co.reshape((n, 3))


def _commit(me, co):
    me.vertices.foreach_set("co", co.ravel())
    me.update()


def profile(obj, stops):
    """Scale local x/y by radius profile along local z (limb shaping).

    stops: [(t, r), ...], t in 0..1 from bottom to top of the mesh.
    """
    import numpy as np
    me, co = _coords(obj)
    zmin, zmax = co[:, 2].min(), co[:, 2].max()
    t = (co[:, 2] - zmin) / max(zmax - zmin, 1e-6)
    r = np.interp(t, [s[0] for s in stops], [s[1] for s in stops])
    co[:, 0] *= r
    co[:, 1] *= r
    _commit(me, co)
    return obj


def bulge_y(obj, zc, width, amt):
    """Push verts outward in local +y/-y around height zc (muscle peak)."""
    import numpy as np
    me, co = _coords(obj)
    w = np.exp(-((co[:, 2] - zc) / width) ** 2)
    co[:, 1] += np.sign(co[:, 1]) * w * amt
    _commit(me, co)
    return obj


def shape_torso(obj):
    """V-taper: narrow waist, broad chest, flattened back."""
    import numpy as np
    me, co = _coords(obj)
    zmin, zmax = co[:, 2].min(), co[:, 2].max()
    t = (co[:, 2] - zmin) / max(zmax - zmin, 1e-6)
    # waist pinch low, chest flare high
    r = 0.86 + 0.28 * t
    co[:, 0] *= r
    co[:, 1] *= 0.9 + 0.15 * t
    # flatten the back plane (local units: mesh spans +/-0.20)
    back = co[:, 1] < -0.14
    co[back, 1] = -0.14 + (co[back, 1] + 0.14) * 0.35
    _commit(me, co)
    return obj


def shape_head(obj):
    """Jawline taper, brow ridge, flattened back of skull, chin.

    NOTE: operates in LOCAL mesh coords (sphere r=0.115 at origin).
    """
    import numpy as np
    me, co = _coords(obj)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    # jaw taper: full effect at local z <= -0.10, none above -0.025
    jaw = np.clip((-0.025 - z) / 0.075, 0, 1)
    x *= 1 - 0.30 * jaw
    # tuck the lower face slightly, push the chin forward
    y -= jaw * 0.011
    chin = np.clip((-0.075 - z) / 0.04, 0, 1) * (y > 0.02)
    y += chin * 0.011
    # brow ridge above the eyes
    brow = np.clip(1 - np.abs(z - 0.045) / 0.018, 0, 1) * (y > 0.055)
    y += brow * 0.009
    # flatten back of skull
    back = y < -0.08
    y[back] = -0.08 + (y[back] + 0.08) * 0.4
    co[:, 0], co[:, 1], co[:, 2] = x, y, z
    _commit(me, co)
    return obj


def shape_fist(obj):
    """Mitten with knuckle ridge and finger taper."""
    import numpy as np
    me, co = _coords(obj)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    # knuckle ridge across the top-front
    ridge = np.clip(1 - np.abs(z - 0.03) / 0.035, 0, 1) * (y > 0.0)
    y += ridge * 0.012
    # taper the pinky side (-x local? fist is symmetric-ish, skip)
    _commit(me, co)
    return obj


# ---------------------------------------------------------------- armature
BONES = {
    # name: (head, tail, parent)
    "root": ((0, 0, 0), (0, 0, 0.25), None),
    "spine": ((0, 0, 1.00), (0, 0, 1.30), "root"),
    "chest": ((0, 0, 1.30), (0, 0, 1.50), "spine"),
    "neck": ((0, 0, 1.50), (0, 0, 1.60), "chest"),
    "head": ((0, 0, 1.60), (0, 0, 1.84), "neck"),
    "upper_arm.L": ((0.20, 0, 1.47), (0.48, 0, 1.47), "chest"),
    "forearm.L": ((0.48, 0, 1.47), (0.74, 0, 1.47), "upper_arm.L"),
    "hand.L": ((0.74, 0, 1.47), (0.90, 0, 1.47), "forearm.L"),
    "upper_arm.R": ((-0.20, 0, 1.47), (-0.48, 0, 1.47), "chest"),
    "forearm.R": ((-0.48, 0, 1.47), (-0.74, 0, 1.47), "upper_arm.R"),
    "hand.R": ((-0.74, 0, 1.47), (-0.90, 0, 1.47), "forearm.R"),
    "thigh.L": ((0.115, 0, 1.00), (0.115, 0, 0.55), "spine"),
    "shin.L": ((0.115, 0, 0.55), (0.115, 0, 0.12), "thigh.L"),
    "foot.L": ((0.115, 0, 0.12), (0.115, 0.24, 0.05), "shin.L"),
    "thigh.R": ((-0.115, 0, 1.00), (-0.115, 0, 0.55), "spine"),
    "shin.R": ((-0.115, 0, 0.55), (-0.115, 0, 0.12), "thigh.R"),
    "foot.R": ((-0.115, 0, 0.12), (-0.115, 0.24, 0.05), "shin.R"),
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
def build_body(S, H, HD, LIPS, SHADE, RED, WHITE, GOLD, DARK):
    X90 = (0, math.radians(90), 0)  # cylinder Z-axis -> X-axis (arms)
    parts = []
    # ---- torso: shaped mass + separate muscle plates ----
    torso = ball("Torso", (0, 0, 1.26), 0.19, S, "spine",
                 scale=(1.0, 0.68, 1.30), subsurf=2, seg=32, rings=24)
    parts.append(shape_torso(torso))
    parts.append(ball("Pelvis", (0, 0, 1.02), 0.155, S, "spine",
                      scale=(1.0, 0.68, 0.75), subsurf=2))
    # pecs
    parts.append(ball("Pec.L", (0.095, 0.070, 1.425), 0.080, S, "chest",
                      scale=(1.15, 0.55, 0.9), subsurf=2))
    parts.append(ball("Pec.R", (-0.095, 0.070, 1.425), 0.080, S, "chest",
                      scale=(1.15, 0.55, 0.9), subsurf=2))
    # ab bumps (4-pack)
    for i, az in enumerate((1.315, 1.235)):
        for j, ax in enumerate((0.052, -0.052)):
            parts.append(ball(f"Abs.{i}{j}", (ax, 0.098, az), 0.042, S,
                              "spine", scale=(1.0, 0.7, 1.15), subsurf=1))
    # lats flare under the arms
    parts.append(ball("Lat.L", (0.175, -0.01, 1.36), 0.075, S, "chest",
                      scale=(0.7, 0.9, 1.3), subsurf=2))
    parts.append(ball("Lat.R", (-0.175, -0.01, 1.36), 0.075, S, "chest",
                      scale=(0.7, 0.9, 1.3), subsurf=2))
    # neck + traps, thick fighter column
    neck = cyl("Neck", (0, 0, 1.560), 0.075, 0.12, S, "neck", subsurf=2)
    parts.append(profile(neck, [(0, 1.15), (0.5, 1.05), (1, 0.95)]))
    parts.append(ball("Trap.L", (0.135, -0.01, 1.525), 0.085, S, "chest",
                      scale=(1.3, 0.8, 0.55), subsurf=2))
    parts.append(ball("Trap.R", (-0.135, -0.01, 1.525), 0.085, S, "chest",
                      scale=(1.3, 0.8, 0.55), subsurf=2))
    # ---- sculpted head ----
    head = ball("Head", (0, 0.005, 1.705), 0.115, S, "head",
                scale=(0.95, 1.05, 1.1), subsurf=2, seg=32, rings=24)
    parts.append(shape_head(head))
    parts.append(ball("NoseBridge", (0, 0.108, 1.700), 0.020, S, "head",
                      scale=(0.7, 0.9, 1.6), subsurf=1))
    parts.append(ball("NoseTip", (0, 0.112, 1.664), 0.020, S, "head",
                      scale=(1.1, 0.9, 0.9), subsurf=1))
    parts.append(ball("Ear.L", (0.104, -0.005, 1.700), 0.030, S, "head",
                      scale=(0.45, 0.8, 1.1), subsurf=1))
    parts.append(ball("Ear.R", (-0.104, -0.005, 1.700), 0.030, S, "head",
                      scale=(0.45, 0.8, 1.1), subsurf=1))
    parts.append(ball("Eye.L", (0.046, 0.102, 1.728), 0.018, DARK, "head"))
    parts.append(ball("Eye.R", (-0.046, 0.102, 1.728), 0.018, DARK, "head"))
    parts.append(cube("Brow.L", (0.054, 0.100, 1.766), (0.030, 0.010, 0.008),
                      DARK, "head"))
    parts.append(cube("Brow.R", (-0.054, 0.100, 1.766), (0.030, 0.010, 0.008),
                      DARK, "head"))
    parts.append(ball("LipUpper", (0, 0.104, 1.652), 0.030, LIPS, "head",
                      scale=(1.15, 0.35, 0.30), subsurf=1))
    parts.append(ball("LipLower", (0, 0.102, 1.643), 0.024, LIPS, "head",
                      scale=(1.05, 0.35, 0.35), subsurf=1))
    # bleached textured crop: rounded top mass + chunky fringe + fades
    parts.append(ball("Hair", (0, -0.015, 1.778), 0.120, H, "head",
                      scale=(1.02, 1.00, 0.55), subsurf=2))
    for i, fx in enumerate((-0.058, -0.020, 0.020, 0.058)):
        parts.append(cube(f"HairChunk.{i}", (fx, 0.090, 1.766),
                          (0.020, 0.018, 0.022), H, "head", subsurf=1))
    parts.append(ball("Fade.L", (0.098, -0.005, 1.735), 0.055, HD, "head",
                      scale=(0.30, 1.1, 0.75), subsurf=1))
    parts.append(ball("Fade.R", (-0.098, -0.005, 1.735), 0.055, HD, "head",
                      scale=(0.30, 1.1, 0.75), subsurf=1))
    parts.append(torus("Chain", (0, 0, 1.550), 0.095, 0.008, GOLD, "chest"))
    # ---- arms: delt caps, tapered limbs, big SF fists ----
    for side, s in (("L", 1), ("R", -1)):
        parts.append(ball(f"Delt.{side}", (s * 0.228, 0, 1.462), 0.075, S,
                          f"upper_arm.{side}", scale=(1.0, 1.0, 1.2),
                          subsurf=2))
        ua = cyl(f"UpperArm.{side}", (s * 0.350, 0, 1.47), 0.068, 0.30, S,
                 f"upper_arm.{side}", X90, subsurf=2)
        parts.append(profile(ua, [(0, 0.80), (0.45, 1.12), (1, 0.95)]))
        bulge_y(ua, 0.02, 0.09, 0.018)  # bicep peak
        fa = cyl(f"Forearm.{side}", (s * 0.615, 0, 1.47), 0.058, 0.28, S,
                 f"forearm.{side}", X90, subsurf=2)
        parts.append(profile(fa, [(0, 0.72), (0.35, 1.05), (1, 0.55)]))
        fist = ball(f"Hand.{side}", (s * 0.845, 0, 1.47), 0.082, S,
                    f"hand.{side}", scale=(1.35, 0.78, 1.0), subsurf=2)
        parts.append(shape_fist(fist))
        parts.append(ball(f"Thumb.{side}", (s * 0.825, 0.055, 1.478), 0.030,
                          S, f"hand.{side}", scale=(1.0, 1.3, 0.8),
                          subsurf=1))
        # red hand wraps over fist + wrist
        wf = cyl(f"WrapFist.{side}", (s * 0.845, 0, 1.47), 0.086, 0.10, RED,
                 f"hand.{side}", X90, subsurf=1)
        parts.append(profile(wf, [(0, 0.92), (0.5, 1.0), (1, 0.92)]))
        ww = cyl(f"WrapWrist.{side}", (s * 0.700, 0, 1.47), 0.058, 0.12, RED,
                 f"forearm.{side}", X90, subsurf=1)
        parts.append(profile(ww, [(0, 1.05), (1, 0.9)]))
    # ---- legs: quad sweep, calf diamonds, heavy feet ----
    for side, s in (("L", 1), ("R", -1)):
        th = cyl(f"Thigh.{side}", (s * 0.115, 0, 0.760), 0.100, 0.46, S,
                 f"thigh.{side}", subsurf=2)
        parts.append(profile(th, [(0, 0.68), (0.45, 1.02), (0.8, 1.18),
                                  (1, 1.05)]))
        sh = cyl(f"Shin.{side}", (s * 0.115, 0, 0.315), 0.062, 0.44, S,
                 f"shin.{side}", subsurf=2)
        parts.append(profile(sh, [(0, 0.55), (0.35, 0.85), (0.62, 1.08),
                                  (1, 0.72)]))
        bulge_y(sh, 0.10, 0.12, 0.020)  # calf diamond
        # foot: heel + wedge + toe cap, all on the foot bone
        parts.append(ball(f"Foot.{side}", (s * 0.115, 0.060, 0.055), 0.075, S,
                          f"foot.{side}", scale=(0.75, 2.0, 0.62), subsurf=2))
        parts.append(ball(f"Toes.{side}", (s * 0.115, 0.210, 0.038), 0.048, S,
                          f"foot.{side}", scale=(0.95, 1.0, 0.62), subsurf=1))
    for side, s in (("L", 1), ("R", -1)):
        aw = cyl(f"AnkleWrap.{side}", (s * 0.115, 0, 0.145), 0.066, 0.10,
                 RED, f"shin.{side}", subsurf=1)
        parts.append(profile(aw, [(0, 1.08), (1, 0.92)]))
    # ---- red Muay Thai shorts, rounded ----
    parts.append(ball("Shorts", (0, 0, 1.000), 0.20, RED, "spine",
                      scale=(0.95, 0.62, 0.60), subsurf=2))
    for side, s in (("L", 1), ("R", -1)):
        parts.append(cyl(f"ShortLeg.{side}", (s * 0.115, 0, 0.825),
                         0.112, 0.20, RED, f"thigh.{side}", subsurf=2))
        parts.append(cyl(f"LegTrim.{side}", (s * 0.115, 0, 0.745),
                         0.116, 0.025, WHITE, f"thigh.{side}"))
    parts.append(cube("Waistband", (0, 0, 1.098), (0.196, 0.146, 0.028),
                      WHITE, "spine", subsurf=1))
    parts.append(cube("Stripe.L", (0.192, 0, 1.000), (0.006, 0.100, 0.110),
                      GOLD, "spine"))
    parts.append(cube("Stripe.R", (-0.192, 0, 1.000), (0.006, 0.100, 0.110),
                      GOLD, "spine"))
    return parts


def build():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    S = mat("Skin", (0.87, 0.66, 0.53), roughness=0.55, subsurface=0.25)
    H = mat("HairBleach", (0.82, 0.79, 0.72), roughness=0.85)
    HD = mat("HairFade", (0.60, 0.56, 0.50), roughness=0.9)
    LIPS = mat("Lips", (0.45, 0.20, 0.18), roughness=0.6)
    SHADE = mat("SkinShade", (0.72, 0.52, 0.42), roughness=0.6)
    RED = mat("ShortsRed", (0.72, 0.05, 0.09), roughness=0.7)
    WHITE = mat("TrimWhite", (0.93, 0.93, 0.94), roughness=0.6)
    GOLD = mat("Gold", (0.83, 0.62, 0.25), roughness=0.35, metallic=0.8)
    DARK = mat("Dark", (0.08, 0.06, 0.05), roughness=0.5)
    build_armature()  # must exist before finish() binds modifiers
    parts = build_body(S, H, HD, LIPS, SHADE, RED, WHITE, GOLD, DARK)
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
