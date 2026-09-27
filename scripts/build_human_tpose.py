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
        # level 1 on 24-32 seg bases is plenty smooth; higher levels only
        # bloat exports (modifiers bake on export) and stall Freestyle
        sm = obj.modifiers.new("Smooth", "SUBSURF")
        sm.levels = min(subsurf, 1)
        sm.render_levels = min(subsurf, 1)
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
        subsurf=0, verts=24, cuts=0):
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius, depth=depth, location=loc, rotation=rotation,
        vertices=verts)
    obj = bpy.context.active_object
    obj.name = name
    if cuts:
        # raw cylinders only have top+bottom rings, so length profiles
        # would only sample the endpoints - cut length rings first
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=cuts)
        bpy.ops.object.mode_set(mode="OBJECT")
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


def sculpt(obj, bumps):
    """Push vertices along a direction with gaussian falloff, in WORLD space.

    bumps: [(center(3), sigma(3), amp, dir(3), mask)] where mask is
    None or (axis, sign) to restrict the effect to one side of the body
    (e.g. front-only pec bulges). This is how muscle definition is built
    into a single smooth mesh instead of stacking visible balls.
    """
    import numpy as np
    me = obj.data
    mw = np.array(obj.matrix_world)
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    wco = co @ mw[:3, :3].T + mw[:3, 3]
    axis_n = np.array([0.0, 0.0, 0.0])
    for center, sigma, amp, direction, mask in bumps:
        w = np.exp(-(((wco[:, 0] - center[0]) / sigma[0]) ** 2
                     + ((wco[:, 1] - center[1]) / sigma[1]) ** 2
                     + ((wco[:, 2] - center[2]) / sigma[2]) ** 2))
        if mask is not None:
            ax, sgn = mask
            w = w * (np.sign(wco[:, ax]) == sgn)
        wco += (w * amp)[:, None] * np.array(direction)
    inv = np.linalg.inv(mw)
    co = wco @ inv[:3, :3].T + inv[:3, 3]
    me.vertices.foreach_set("co", co.ravel())
    me.update()
    return obj


def shape_axis(obj, xstops, ystops):
    """Independently scale local x and y along local z.

    stops: [(t, mult), ...] with t 0..1 bottom->top. Gives real torso
    silhouettes (wide shoulders, pinched waist) instead of plain
    ellipsoids.
    """
    import numpy as np
    me, co = _coords(obj)
    zmin, zmax = co[:, 2].min(), co[:, 2].max()
    t = (co[:, 2] - zmin) / max(zmax - zmin, 1e-6)
    rx = np.interp(t, [s[0] for s in xstops], [s[1] for s in xstops])
    ry = np.interp(t, [s[0] for s in ystops], [s[1] for s in ystops])
    co[:, 0] *= rx
    co[:, 1] *= ry
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
    "upper_arm.L": ((0.20, 0, 1.50), (0.49, 0, 1.50), "chest"),
    "forearm.L": ((0.49, 0, 1.50), (0.74, 0, 1.50), "upper_arm.L"),
    "hand.L": ((0.74, 0, 1.50), (0.90, 0, 1.50), "forearm.L"),
    "upper_arm.R": ((-0.20, 0, 1.50), (-0.49, 0, 1.50), "chest"),
    "forearm.R": ((-0.49, 0, 1.50), (-0.74, 0, 1.50), "upper_arm.R"),
    "hand.R": ((-0.74, 0, 1.50), (-0.90, 0, 1.50), "forearm.R"),
    "thigh.L": ((0.116, 0, 1.00), (0.116, 0, 0.545), "spine"),
    "shin.L": ((0.116, 0, 0.545), (0.116, 0, 0.115), "thigh.L"),
    "foot.L": ((0.116, 0, 0.115), (0.116, 0.24, 0.045), "shin.L"),
    "thigh.R": ((-0.116, 0, 1.00), (-0.116, 0, 0.545), "spine"),
    "shin.R": ((-0.116, 0, 0.545), (-0.116, 0, 0.115), "thigh.R"),
    "foot.R": ((-0.116, 0, 0.115), (-0.116, 0.24, 0.045), "shin.R"),
}


# ---------------------------------------------------------------- ink + dirt
def apply_dirt(parts):
    """Bake crevice darkening into a white-initialized 'Dirt' color
    attribute. three.js multiplies COLOR_0 over base color automatically,
    so the web demo and any PBR viewer get cheap SF-style grime. Safe
    fallback: attribute stays white (no darkening) if the op is missing.
    """
    for o in parts:
        me = o.data
        attr = me.color_attributes.get("Dirt")
        if attr is None:
            attr = me.color_attributes.new("Dirt", "FLOAT_COLOR", "POINT")
        n = len(me.vertices)
        attr.data.foreach_set("color", [1.0, 1.0, 1.0, 1.0] * n)
        me.color_attributes.active_color = attr
        try:
            me.color_attributes.render_color_index = (
                me.color_attributes.find("Dirt"))
        except Exception:  # noqa: BLE001 - older Blender, active is enough
            pass
    for o in parts:
        try:
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.mode_set(mode="VERTEX_PAINT")
            bpy.ops.paint.vertex_color_dirt()
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception as e:  # noqa: BLE001 - keep whites, skip dirt
            print(f"[dirt] skipped {o.name}: {e}")
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:  # noqa: BLE001
                pass


def build_ink(parts):
    """Inverted-hull ink outlines, SFIV style. Hulls copy the part's
    armature bind so they follow poses. They carry NO slicer-facing
    role: excluded from the FBX (game) export by selection, included in
    the GLB (web demo) where the viewer renders them BackSide.
    """
    ink_mat = mat("Ink", (0.015, 0.015, 0.02), roughness=1.0)
    # cull the hull's outer faces: with flipped normals this leaves only
    # the inner far-side shell visible = classic outline rim
    ink_mat.use_backface_culling = True
    arm = bpy.data.objects["FighterArmature"]
    hulls = []
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        h = o.copy()
        h.data = o.data.copy()
        h.name = "Ink_" + o.name
        bpy.context.collection.objects.link(h)
        while h.vertex_groups:
            h.vertex_groups.remove(h.vertex_groups[0])
        for m in h.modifiers:
            if m.type == "SUBSURF":
                m.levels = 1
                m.render_levels = 1
            elif m.type == "ARMATURE":
                m.object = arm
        if not any(m.type == "ARMATURE" for m in h.modifiers):
            mod = h.modifiers.new("ArmatureBind", "ARMATURE")
            mod.object = arm
        h.select_set(True)
        bpy.context.view_layer.objects.active = h
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        h.select_set(False)
        vg = h.vertex_groups.new(name=o.vertex_groups[0].name)
        vg.add(list(range(len(h.data.vertices))), 1.0, "REPLACE")
        h.data.materials.clear()
        h.data.materials.append(ink_mat)
        sc = h.scale
        h.scale = (sc[0] * 1.03, sc[1] * 1.03, sc[2] * 1.03)
        # stills default to Freestyle lines (see preview.py); hulls serve
        # the realtime GLB where the viewer draws them BackSide
        h.hide_render = True
        hulls.append(h)
    return hulls


def select_for_export(parts, hulls, with_ink):
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts + (hulls if with_ink else []):
        o.select_set(True)
    bpy.data.objects["FighterArmature"].select_set(True)
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
    # ---- torso: one smooth mesh, muscles sculpted in ----
    torso = ball("Torso", (0, 0, 1.245), 0.205, S, "spine",
                 scale=(1.0, 0.74, 1.45), subsurf=2, seg=32, rings=24)
    shape_axis(torso,
               [(0, 0.92), (0.22, 0.82), (0.46, 0.78), (0.66, 0.95),
                (0.86, 1.12), (1.0, 1.16)],
               [(0, 1.00), (0.25, 0.86), (0.5, 0.82), (0.78, 0.98), (1.0, 1.00)])
    front = (1, 1)
    back = (1, -1)
    bumps = [
        # pectorals + sternum groove
        ((0.100, 0, 1.400), (0.086, 0.60, 0.088), 0.038, (0, 1, 0), front),
        ((-0.100, 0, 1.400), (0.086, 0.60, 0.088), 0.038, (0, 1, 0), front),
        ((0.000, 0, 1.395), (0.024, 0.60, 0.115), -0.026, (0, 1, 0), front),
        # six-pack + linea alba
        ((0.052, 0, 1.325), (0.038, 0.60, 0.042), 0.026, (0, 1, 0), front),
        ((-0.052, 0, 1.325), (0.038, 0.60, 0.042), 0.026, (0, 1, 0), front),
        ((0.052, 0, 1.245), (0.038, 0.60, 0.042), 0.024, (0, 1, 0), front),
        ((-0.052, 0, 1.245), (0.038, 0.60, 0.042), 0.024, (0, 1, 0), front),
        ((0.052, 0, 1.168), (0.038, 0.60, 0.040), 0.020, (0, 1, 0), front),
        ((-0.052, 0, 1.168), (0.038, 0.60, 0.040), 0.020, (0, 1, 0), front),
        ((0.000, 0, 1.245), (0.015, 0.60, 0.130), -0.020, (0, 1, 0), front),
        # lat flare
        ((0.185, 0, 1.350), (0.060, 0.60, 0.160), 0.019, (1, 0, 0), (0, 1)),
        ((-0.185, 0, 1.350), (0.060, 0.60, 0.160), 0.019, (-1, 0, 0), (0, -1)),
        # spine groove
        ((0.000, 0, 1.330), (0.024, 0.60, 0.220), 0.022, (0, 1, 0), back),
    ]
    parts.append(sculpt(torso, bumps))
    # hips / glutes, bridging into the thighs
    hip = ball("Pelvis", (0, 0, 1.020), 0.170, S, "spine",
               scale=(1.0, 0.74, 0.74), subsurf=2)
    parts.append(shape_axis(hip,
                            [(0, 0.90), (0.35, 1.0), (1.0, 0.98)],
                            [(0, 0.94), (0.4, 1.06), (1.0, 0.98)]))
    # neck + traps
    neck = cyl("Neck", (0, 0, 1.540), 0.082, 0.14, S, "neck", subsurf=2,
               cuts=3)
    parts.append(profile(neck, [(0, 1.20), (0.5, 1.08), (1, 0.98)]))
    parts.append(ball("Trap.L", (0.128, -0.012, 1.504), 0.100, S, "chest",
                      scale=(1.42, 0.90, 0.56), subsurf=2))
    parts.append(ball("Trap.R", (-0.128, -0.012, 1.504), 0.100, S, "chest",
                      scale=(1.42, 0.90, 0.56), subsurf=2))
    # ---- head ----
    head = ball("Head", (0, 0.006, 1.700), 0.134, S, "head",
                scale=(0.92, 1.02, 1.05), subsurf=2, seg=32, rings=24)
    parts.append(shape_head(head))
    parts.append(ball("NoseBridge", (0, 0.128, 1.696), 0.020, S, "head",
                      scale=(0.74, 0.90, 1.50), subsurf=1))
    parts.append(ball("NoseTip", (0, 0.132, 1.660), 0.024, S, "head",
                      scale=(1.05, 0.90, 0.88), subsurf=1))
    parts.append(ball("NoseWing.L", (0.022, 0.122, 1.658), 0.016, S, "head",
                      scale=(1.0, 0.85, 0.85), subsurf=1))
    parts.append(ball("NoseWing.R", (-0.022, 0.122, 1.658), 0.016, S, "head",
                      scale=(1.0, 0.85, 0.85), subsurf=1))
    parts.append(ball("Ear.L", (0.111, -0.006, 1.695), 0.032, S, "head",
                      scale=(0.44, 0.80, 1.10), subsurf=1))
    parts.append(ball("Ear.R", (-0.111, -0.006, 1.695), 0.032, S, "head",
                      scale=(0.44, 0.80, 1.10), subsurf=1))
    parts.append(ball("Eye.L", (0.049, 0.124, 1.724), 0.020, DARK, "head",
                      scale=(1.30, 0.85, 0.58)))
    parts.append(ball("Eye.R", (-0.049, 0.124, 1.724), 0.020, DARK, "head",
                      scale=(1.30, 0.85, 0.58)))
    parts.append(cube("Brow.L", (0.058, 0.124, 1.752), (0.030, 0.009, 0.007),
                      DARK, "head", subsurf=1))
    parts.append(cube("Brow.R", (-0.058, 0.124, 1.752), (0.030, 0.009, 0.007),
                      DARK, "head", subsurf=1))
    parts.append(ball("LipUpper", (0, 0.126, 1.646), 0.028, LIPS, "head",
                      scale=(1.05, 0.32, 0.26), subsurf=1))
    parts.append(ball("LipLower", (0, 0.124, 1.638), 0.022, LIPS, "head",
                      scale=(0.95, 0.32, 0.30), subsurf=1))
    # bleached crop sitting ON the skull, clear of the brow
    parts.append(ball("Hair", (0, -0.016, 1.802), 0.128, H, "head",
                      scale=(1.02, 1.01, 0.54), subsurf=2))
    for i, (fx, fz, fw) in enumerate((
            (-0.070, 1.774, 0.026), (-0.026, 1.782, 0.030),
            (0.026, 1.780, 0.030), (0.070, 1.772, 0.026))):
        parts.append(cube(f"HairFringe.{i}", (fx, 0.098, fz),
                          (fw, 0.026, 0.040), H, "head", subsurf=1))
    parts.append(ball("HairSweep", (0.030, 0.026, 1.840), 0.076, H, "head",
                      scale=(0.95, 0.85, 0.36), subsurf=1))
    parts.append(torus("Chain", (0, 0, 1.522), 0.094, 0.008, GOLD, "chest"))
    # ---- arms: sculpted bicep/tricep, tapered forearm, big fists ----
    for side, s in (("L", 1), ("R", -1)):
        parts.append(ball(f"Delt.{side}", (s * 0.218, 0, 1.486), 0.100, S,
                          f"upper_arm.{side}", scale=(1.0, 1.0, 1.04),
                          subsurf=2))
        ua = cyl(f"UpperArm.{side}", (s * 0.360, 0, 1.500), 0.088, 0.28, S,
                 f"upper_arm.{side}", X90, subsurf=2, cuts=6)
        parts.append(profile(ua, [(0, 0.84), (0.45, 1.10), (1, 0.94)]))
        parts.append(sculpt(ua, [
            ((s * 0.355, 0, 1.500), (0.095, 0.60, 0.055), 0.030,
             (0, 1, 0), front),   # bicep
            ((s * 0.360, 0, 1.500), (0.100, 0.60, 0.060), 0.022,
             (0, -1, 0), back),   # tricep
        ]))
        parts.append(ball(f"Elbow.{side}", (s * 0.496, 0, 1.500), 0.080, S,
                          f"forearm.{side}", subsurf=2))
        fa = cyl(f"Forearm.{side}", (s * 0.626, 0, 1.500), 0.075, 0.25, S,
                 f"forearm.{side}", X90, subsurf=2, cuts=6)
        parts.append(profile(fa, [(0, 0.70), (0.35, 1.02), (1, 0.58)]))
        parts.append(sculpt(fa, [
            ((s * 0.560, 0, 1.500), (0.055, 0.60, 0.055), 0.020,
             (0, 1, 0), front),   # brachioradialis
        ]))
        fist = ball(f"Hand.{side}", (s * 0.858, 0, 1.500), 0.096, S,
                    f"hand.{side}", scale=(1.28, 0.86, 1.06), subsurf=2)
        parts.append(shape_fist(fist))
        parts.append(ball(f"Thumb.{side}", (s * 0.838, 0.062, 1.512), 0.034,
                          S, f"hand.{side}", scale=(1.0, 1.30, 0.84),
                          subsurf=1))
        wf = cyl(f"WrapFist.{side}", (s * 0.858, 0, 1.500), 0.101, 0.10, RED,
                 f"hand.{side}", X90, subsurf=1, cuts=2)
        parts.append(profile(wf, [(0, 0.94), (0.5, 1.0), (1, 0.94)]))
        ww = cyl(f"WrapWrist.{side}", (s * 0.730, 0, 1.500), 0.082, 0.11, RED,
                 f"forearm.{side}", X90, subsurf=1, cuts=3)
        parts.append(profile(ww, [(0, 1.04), (1, 0.90)]))
    # ---- legs: sculpted quads/calves, heavy feet ----
    for side, s in (("L", 1), ("R", -1)):
        parts.append(ball(f"HipJoint.{side}", (s * 0.122, 0, 1.000), 0.132, S,
                          f"thigh.{side}", scale=(1.0, 0.96, 0.88),
                          subsurf=2))
        th = cyl(f"Thigh.{side}", (s * 0.120, 0, 0.780), 0.122, 0.45, S,
                 f"thigh.{side}", subsurf=2, cuts=6)
        parts.append(profile(th, [(0, 0.74), (0.45, 1.00), (0.82, 1.10),
                                  (1, 0.98)]))
        parts.append(sculpt(th, [
            ((s * 0.120, 0, 0.790), (0.090, 0.60, 0.120), 0.026,
             (0, 1, 0), front),   # quad
            ((s * 0.120, 0, 0.820), (0.085, 0.60, 0.115), 0.018,
             (0, -1, 0), back),   # hamstring
        ]))
        parts.append(ball(f"Knee.{side}", (s * 0.120, 0, 0.552), 0.086, S,
                          f"shin.{side}", subsurf=2))
        sh = cyl(f"Shin.{side}", (s * 0.120, 0, 0.335), 0.080, 0.42, S,
                 f"shin.{side}", subsurf=2, cuts=6)
        parts.append(profile(sh, [(0, 0.56), (0.32, 0.88), (0.60, 1.08),
                                  (1, 0.78)]))
        parts.append(sculpt(sh, [
            ((s * 0.120, 0, 0.335), (0.070, 0.60, 0.115), 0.034,
             (0, -1, 0), back),   # calf
            ((s * 0.120, 0, 0.300), (0.060, 0.60, 0.110), 0.014,
             (0, 1, 0), front),   # tibialis
        ]))
        parts.append(ball(f"Foot.{side}", (s * 0.120, 0.058, 0.052), 0.080, S,
                          f"foot.{side}", scale=(0.82, 1.95, 0.60),
                          subsurf=2))
        parts.append(ball(f"Toes.{side}", (s * 0.120, 0.212, 0.036), 0.050, S,
                          f"foot.{side}", scale=(1.00, 1.05, 0.60),
                          subsurf=1))
        aw = cyl(f"AnkleWrap.{side}", (s * 0.120, 0, 0.140), 0.084, 0.10,
                 RED, f"shin.{side}", subsurf=1, cuts=2)
        parts.append(profile(aw, [(0, 1.04), (1, 0.90)]))
    # ---- Muay Thai shorts: loose, high-cut, mid-thigh ----
    # trunk must be wider than hip joints (0.254) or the thighs punch through
    trunk = ball("ShortsTrunk", (0, 0, 1.010), 0.272, RED, "spine",
                 scale=(1.0, 0.72, 0.60), subsurf=2)
    parts.append(shape_axis(trunk,
                            [(0, 0.96), (0.5, 1.0), (1.0, 0.86)],
                            [(0, 1.00), (0.5, 1.0), (1.0, 0.84)]))
    for side, s in (("L", 1), ("R", -1)):
        leg = cyl(f"ShortLeg.{side}", (s * 0.130, 0, 0.855), 0.168, 0.36,
                  RED, f"thigh.{side}", subsurf=2, cuts=4)
        parts.append(profile(leg, [(0, 1.12), (0.45, 1.02), (1, 0.88)]))
    # waistband hugs the trunk instead of sitting on it like a plate
    band = ball("Waistband", (0, 0, 1.150), 0.250, WHITE, "spine",
                scale=(1.0, 0.72, 0.075), subsurf=2)
    parts.append(shape_axis(band, [(0, 1.0), (1, 1.0)],
                            [(0, 1.0), (1, 1.0)]))
    return parts


def build(skip_dirt=False, skip_ink=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    S = mat("Skin", (0.87, 0.66, 0.53), roughness=0.55, subsurface=0.25)
    H = mat("HairBleach", (0.86, 0.79, 0.60), roughness=0.85)
    HD = mat("HairFade", (0.60, 0.56, 0.50), roughness=0.9)
    LIPS = mat("Lips", (0.45, 0.20, 0.18), roughness=0.6)
    SHADE = mat("SkinShade", (0.72, 0.52, 0.42), roughness=0.6)
    RED = mat("ShortsRed", (0.72, 0.05, 0.09), roughness=0.7)
    WHITE = mat("TrimWhite", (0.93, 0.93, 0.94), roughness=0.6)
    GOLD = mat("Gold", (0.83, 0.62, 0.25), roughness=0.35, metallic=0.8)
    DARK = mat("Dark", (0.08, 0.06, 0.05), roughness=0.5)
    build_armature()  # must exist before finish() binds modifiers
    parts = build_body(S, H, HD, LIPS, SHADE, RED, WHITE, GOLD, DARK)
    if skip_dirt:
        print("[build] dirt skipped (fast mode)")
    else:
        apply_dirt(parts)
    hulls = [] if skip_ink else build_ink(parts)
    print(f"[build] fighter done: {len(parts)} parts + {len(hulls)} ink hulls"
          f" + armature, {len(BONES)} bones")
    return parts, hulls


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--fbx", default=None)
    p.add_argument("--glb", default=None)
    p.add_argument("--skip-dirt", action="store_true",
                   help="skip baked crevice shading (fast iteration)")
    p.add_argument("--skip-ink", action="store_true",
                   help="skip ink hulls (fast iteration)")
    return p.parse_args(argv)


def main(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    args = parse_args(argv)
    if bpy is None:
        raise RuntimeError("Run inside Blender: blender --background --python "
                           "scripts/build_human_tpose.py")
    build(skip_dirt=args.skip_dirt, skip_ink=args.skip_ink)
    parts = [o for o in bpy.data.objects if o.type == "MESH"
             and not o.name.startswith("Ink_")]
    hulls = [o for o in bpy.data.objects if o.type == "MESH"
             and o.name.startswith("Ink_")]
    bpy.ops.wm.save_as_mainfile(filepath=args.out)
    print(f"[build] saved {args.out}")
    if args.fbx:
        # game export: ink excluded so slicer sprites stay clean
        select_for_export(parts, hulls, with_ink=False)
        bpy.ops.export_scene.fbx(filepath=args.fbx, use_selection=True,
                                 add_leaf_bones=False)
        print(f"[build] exported {args.fbx}")
    if args.glb:
        # web export: ink included, viewer renders it BackSide.
        # ACTIVE forces the Dirt color attribute out as COLOR_0.
        select_for_export(parts, hulls, with_ink=True)
        bpy.ops.export_scene.gltf(filepath=args.glb, export_format="GLB",
                                  use_selection=True,
                                  export_vertex_color="ACTIVE")
        print(f"[build] exported {args.glb}")


if __name__ == "__main__":
    main(sys.argv)
