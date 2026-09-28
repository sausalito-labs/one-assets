"""Build the Takeru fighter: T-pose, continuous body, SF-style musculature.

Approach: a dense anatomical skeleton (52 stations, each with its own
elliptical cross-section) is skinned into ONE manifold mesh by Blender's
Skin modifier, then muscle groups are sculpted along the surface normals
with explicit grooves between them. Dense stations are what let the form
carry anatomy; grooves are what make it read as muscle instead of an
inflated balloon.

Usage (inside the `3d` workflow shell):
    blender --background --python scripts/build_fighter.py -- \
        --out characters/takeru/takeru.blend \
        --glb demo/fighter/fighter.glb \
        --face reference/photo.jpg
"""
import argparse
import math
import sys

import numpy as np

try:
    import bpy
    from mathutils import Vector
except ImportError:
    bpy = None


# ------------------------------------------------------------------ materials
def mat(name, color, roughness=0.6, metallic=0.0):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes.get("Principled BSDF")
    b.inputs["Base Color"].default_value = (*color, 1.0)
    b.inputs["Roughness"].default_value = roughness
    if "Metallic" in b.inputs:
        b.inputs["Metallic"].default_value = metallic
    return m


# ------------------------------------------------------------------ primitives
def _finish(obj, material, bone, subsurf=0):
    for p in obj.data.polygons:
        p.use_smooth = True
    if material:
        obj.data.materials.append(material)
    arm = bpy.data.objects.get("FighterArmature")
    if arm:
        m = obj.modifiers.new("Bind", "ARMATURE")
        m.object = arm
    if subsurf:
        s = obj.modifiers.new("Smooth", "SUBSURF")
        s.levels = s.render_levels = min(subsurf, 2)
    if bone:
        vg = obj.vertex_groups.new(name=bone)
        vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")
    return obj


def cube(name, loc, scale, material, bone, subsurf=0):
    bpy.ops.mesh.primitive_cube_add(size=2.0, location=loc, scale=scale)
    o = bpy.context.active_object
    o.name = name
    return _finish(o, material, bone, subsurf)


def ball(name, loc, radius, material, bone, scale=(1, 1, 1), subsurf=0,
         seg=32, rings=24):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=radius, location=loc,
                                         scale=scale, segments=seg,
                                         ring_count=rings)
    o = bpy.context.active_object
    o.name = name
    return _finish(o, material, bone, subsurf)


def cyl(name, loc, radius, depth, material, bone, rotation=(0, 0, 0),
        subsurf=0, verts=24, cuts=0):
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=depth,
                                        location=loc, rotation=rotation,
                                        vertices=verts)
    o = bpy.context.active_object
    o.name = name
    if cuts:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.subdivide(number_cuts=cuts)
        bpy.ops.object.mode_set(mode="OBJECT")
    return _finish(o, material, bone, subsurf)


def torus(name, loc, major, minor, material, bone):
    bpy.ops.mesh.primitive_torus_add(location=loc, major_radius=major,
                                     minor_radius=minor, major_segments=24,
                                     minor_segments=10)
    o = bpy.context.active_object
    o.name = name
    return _finish(o, material, bone)


# ------------------------------------------------------------------ shaping
def _co(obj):
    me = obj.data
    n = len(me.vertices)
    a = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", a)
    return me, a.reshape(-1, 3)


def _put(me, co):
    me.vertices.foreach_set("co", co.ravel())
    me.update()


def normals_of(me):
    """Per-vertex normals from polygons (vertex_normals.foreach_get is
    unreliable right after modifier_apply)."""
    n = len(me.vertices)
    acc = np.zeros((n, 3))
    pn = np.empty(len(me.polygons) * 3, dtype=np.float64)
    me.polygons.foreach_get("normal", pn)
    pn = pn.reshape(-1, 3)
    for i, p in enumerate(me.polygons):
        for v in p.vertices:
            acc[v] += pn[i]
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return acc / ln


def sculpt(obj, bumps):
    """bumps: (center, (sigma_x, sigma_z), amp, direction[, mask]).

    Displaces along each vertex's own normal, weighted by how much it faces
    `direction`. Positive amp = muscle belly, negative = groove between
    muscle groups.
    """
    me, co = _co(obj)
    mw = np.array(obj.matrix_world)
    wn = normals_of(me)
    wco = co @ mw[:3, :3].T + mw[:3, 3]
    nrm = np.linalg.inv(mw[:3, :3]).T
    wn = wn @ nrm.T
    wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-9)
    inv = np.linalg.inv(mw[:3, :3])
    disp = np.zeros_like(co)
    for bump in bumps:
        c, sig, amp, direction = bump[:4]
        d = np.array(direction, float)
        d /= np.linalg.norm(d)
        w = np.exp(-(((wco[:, 0] - c[0]) / sig[0]) ** 2
                     + ((wco[:, 2] - c[2]) / sig[1]) ** 2))
        w = w * np.clip(wn @ d, 0.0, 1.0)
        disp += (wn * (w * amp)[:, None]) @ inv.T
    _put(me, co + disp)
    print(f"[sculpt] {obj.name}: max {np.linalg.norm(disp, axis=1).max():.4f}")
    return obj


def shape_axis(obj, xs, ys):
    me, co = _co(obj)
    z0, z1 = co[:, 2].min(), co[:, 2].max()
    t = (co[:, 2] - z0) / max(z1 - z0, 1e-6)
    co[:, 0] *= np.interp(t, [a for a, _ in xs], [b for _, b in xs])
    co[:, 1] *= np.interp(t, [a for a, _ in ys], [b for _, b in ys])
    _put(me, co)
    return obj


def flatten_sole(obj, z=0.018):
    me, co = _co(obj)
    mw = np.array(obj.matrix_world)
    w = co @ mw[:3, :3].T + mw[:3, 3]
    w[w[:, 2] < z, 2] = z
    inv = np.linalg.inv(mw)
    _put(me, w @ inv[:3, :3].T + inv[:3, 3])
    return obj


# ------------------------------------------------------------------ skeleton
# (name, x, y, z, rx, ry, parent)
S = 1
SK = [
    ("pelvis", 0, 0, 0.980, 0.150, 0.112, None),
    ("hips",   0, 0, 1.020, 0.152, 0.114, "pelvis"),
    ("waist",  0, 0, 1.160, 0.116, 0.092, "hips"),
    ("ribs",   0, 0, 1.290, 0.134, 0.100, "waist"),
    ("chest",  0, 0, 1.405, 0.158, 0.108, "ribs"),
    ("uchest", 0, 0, 1.472, 0.166, 0.104, "chest"),
    ("neck",   0, 0, 1.560, 0.070, 0.066, "uchest"),
    ("head",   0, 0, 1.660, 0.062, 0.060, "neck"),
]
for s, side in ((1, "L"), (-1, "R")):
    SK += [
        (f"sh{side}",   s * 0.175, 0, 1.495, 0.088, 0.088, "uchest"),
        (f"delt{side}", s * 0.238, 0, 1.492, 0.100, 0.096, f"sh{side}"),
        (f"bicep{side}", s * 0.330, 0, 1.490, 0.086, 0.082, f"delt{side}"),
        (f"midarm{side}", s * 0.415, 0, 1.490, 0.070, 0.068, f"bicep{side}"),
        (f"elb{side}",  s * 0.490, 0, 1.490, 0.058, 0.058, f"midarm{side}"),
        (f"fup{side}",  s * 0.560, 0, 1.490, 0.068, 0.064, f"elb{side}"),
        (f"fore{side}", s * 0.640, 0, 1.490, 0.058, 0.054, f"fup{side}"),
        (f"wrist{side}", s * 0.728, 0, 1.490, 0.043, 0.042, f"fore{side}"),
        (f"fist{side}", s * 0.800, 0, 1.490, 0.066, 0.060, f"wrist{side}"),
        (f"knuck{side}", s * 0.862, 0, 1.490, 0.058, 0.052, f"fist{side}"),
        (f"hip{side}",   s * 0.112, 0, 0.980, 0.092, 0.092, "pelvis"),
        (f"thup{side}",  s * 0.112, 0, 0.880, 0.096, 0.096, f"hip{side}"),
        (f"thigh{side}", s * 0.112, 0, 0.760, 0.080, 0.078, f"thup{side}"),
        (f"thlo{side}",  s * 0.112, 0, 0.620, 0.080, 0.078, f"thigh{side}"),
        (f"knee{side}",  s * 0.112, 0, 0.530, 0.068, 0.068, f"thlo{side}"),
        (f"calfup{side}", s * 0.112, 0, 0.430, 0.074, 0.072, f"knee{side}"),
        (f"calf{side}",  s * 0.112, 0, 0.330, 0.068, 0.064, f"calfup{side}"),
        (f"calfdn{side}", s * 0.112, 0, 0.220, 0.048, 0.047, f"calf{side}"),
        (f"ankle{side}", s * 0.112, 0, 0.120, 0.042, 0.041, f"calfdn{side}"),
        (f"heel{side}",  s * 0.112, -0.030, 0.048, 0.044, 0.048, f"ankle{side}"),
        (f"ball{side}",  s * 0.112, 0.062, 0.040, 0.046, 0.046, f"heel{side}"),
        (f"toes{side}",  s * 0.112, 0.128, 0.030, 0.042, 0.037, f"ball{side}"),
    ]


def build_body(skin_mat):
    me = bpy.data.meshes.new("Body")
    obj = bpy.data.objects.new("Body", me)
    bpy.context.collection.objects.link(obj)
    import bmesh
    bm = bmesh.new()
    idx = {}
    for i, (n, x, y, z, rx, ry, par) in enumerate(SK):
        idx[n] = bm.verts.new((x, y, z))
    bm.verts.ensure_lookup_table()
    for i, (n, x, y, z, rx, ry, par) in enumerate(SK):
        if par:
            bm.edges.new((idx[par], idx[n]))
    bm.to_mesh(me)
    bm.free()
    obj.modifiers.new("Skin", "SKIN")
    for i, (n, x, y, z, rx, ry, par) in enumerate(SK):
        me.skin_vertices[0].data[i].radius = (rx, ry)
    me.skin_vertices[0].data[0].use_root = True
    sub = obj.modifiers.new("Sub", "SUBSURF")
    sub.levels = sub.render_levels = 3
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Skin")
    bpy.ops.object.modifier_apply(modifier="Sub")
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.materials.append(skin_mat)
    sculpt(obj, MUSCLE)
    flatten_sole(obj, 0.018)
    return obj


F = (0, 1, 0)      # front
B = (0, -1, 0)     # back
L = (1, 0, 0)      # character's left / screen right
R = (-1, 0, 0)

# (center, (sigma_x, sigma_z), amp, facing)
MUSCLE = [
    # chest
    ((0.092, 0, 1.430), (0.064, 0.046), 0.044, F),
    ((-0.092, 0, 1.430), (0.064, 0.046), 0.044, F),
    ((0.000, 0, 1.420), (0.011, 0.058), -0.040, F),
    ((0.000, 0, 1.386), (0.076, 0.010), -0.032, F),
    ((0.150, 0, 1.440), (0.024, 0.040), -0.028, F),
    ((-0.150, 0, 1.440), (0.024, 0.040), -0.028, F),
    # abs
    ((0.046, 0, 1.312), (0.028, 0.026), 0.038, F),
    ((-0.046, 0, 1.312), (0.028, 0.026), 0.038, F),
    ((0.046, 0, 1.238), (0.028, 0.026), 0.036, F),
    ((-0.046, 0, 1.238), (0.028, 0.026), 0.036, F),
    ((0.046, 0, 1.166), (0.028, 0.024), 0.030, F),
    ((-0.046, 0, 1.166), (0.028, 0.024), 0.030, F),
    ((0.000, 0, 1.240), (0.008, 0.092), -0.036, F),
    ((0.000, 0, 1.276), (0.052, 0.008), -0.032, F),
    ((0.000, 0, 1.200), (0.052, 0.008), -0.032, F),
    ((0.108, 0, 1.235), (0.022, 0.070), -0.030, F),
    ((-0.108, 0, 1.235), (0.022, 0.070), -0.030, F),
    # back
    ((0.000, 0, 1.320), (0.011, 0.170), -0.034, B),
    ((0.078, 0, 1.500), (0.048, 0.044), 0.042, B),
    ((-0.078, 0, 1.500), (0.048, 0.044), 0.042, B),
    ((0.070, 0, 1.320), (0.040, 0.080), 0.038, B),
    ((-0.070, 0, 1.320), (0.040, 0.080), 0.038, B),
    ((0.098, 0, 0.975), (0.066, 0.072), 0.048, B),
    ((-0.098, 0, 0.975), (0.066, 0.072), 0.048, B),
    ((0.000, 0, 0.980), (0.011, 0.064), -0.030, B),
    # deltoid + arm
    ((0.246, 0, 1.492), (0.060, 0.072), 0.058, L),
    ((-0.246, 0, 1.492), (0.060, 0.072), 0.058, R),
    ((0.300, 0, 1.492), (0.018, 0.066), -0.026, L),
    ((-0.300, 0, 1.492), (0.018, 0.066), -0.026, R),
    ((0.336, 0, 1.492), (0.070, 0.042), 0.052, F),
    ((-0.336, 0, 1.492), (0.070, 0.042), 0.052, F),
    ((0.344, 0, 1.492), (0.074, 0.046), 0.040, B),
    ((-0.344, 0, 1.492), (0.074, 0.046), 0.040, B),
    ((0.338, 0, 1.516), (0.068, 0.014), -0.022, F),
    ((-0.338, 0, 1.516), (0.068, 0.014), -0.022, F),
    ((0.338, 0, 1.466), (0.068, 0.014), -0.020, F),
    ((-0.338, 0, 1.466), (0.068, 0.014), -0.020, F),
    ((0.470, 0, 1.492), (0.020, 0.058), -0.024, F),
    ((-0.470, 0, 1.492), (0.020, 0.058), -0.024, F),
    ((0.566, 0, 1.492), (0.046, 0.046), 0.026, F),
    ((-0.566, 0, 1.492), (0.046, 0.046), 0.026, F),
    ((0.580, 0, 1.492), (0.050, 0.048), 0.018, B),
    ((-0.580, 0, 1.492), (0.050, 0.048), 0.018, B),
    # legs
    ((0.104, 0, 0.820), (0.072, 0.098), 0.034, F),      # rectus
    ((-0.104, 0, 0.820), (0.072, 0.098), 0.034, F),
    ((0.150, 0, 0.760), (0.038, 0.082), 0.030, F),      # vastus lat
    ((-0.150, 0, 0.760), (0.038, 0.082), 0.030, F),
    ((0.062, 0, 0.618), (0.034, 0.056), 0.026, F),      # vastus med
    ((-0.062, 0, 0.618), (0.034, 0.056), 0.026, F),
    ((0.112, 0, 0.800), (0.016, 0.130), -0.028, F),     # quad split
    ((-0.112, 0, 0.800), (0.016, 0.130), -0.028, F),
    ((0.190, 0, 0.790), (0.020, 0.120), -0.026, L),     # quad/ham line
    ((-0.190, 0, 0.790), (0.020, 0.120), -0.026, R),
    ((0.112, 0, 0.830), (0.070, 0.094), 0.036, B),      # hamstring
    ((-0.112, 0, 0.830), (0.070, 0.094), 0.036, B),
    ((0.112, 0, 0.420), (0.048, 0.076), 0.056, B),      # gastroc
    ((-0.112, 0, 0.420), (0.048, 0.076), 0.056, B),
    ((0.112, 0, 0.360), (0.018, 0.052), -0.024, B),     # calf split
    ((-0.112, 0, 0.360), (0.018, 0.052), -0.024, B),
    ((0.112, 0, 0.300), (0.044, 0.086), 0.018, F),      # tibialis
    ((-0.112, 0, 0.300), (0.044, 0.086), 0.018, F),
]


# ------------------------------------------------------------------ armature
BONES = {
    "root": ((0, 0, 0), (0, 0, 0.25), None),
    "spine": ((0, 0, 1.00), (0, 0, 1.30), "root"),
    "chest": ((0, 0, 1.30), (0, 0, 1.50), "spine"),
    "neck": ((0, 0, 1.50), (0, 0, 1.60), "chest"),
    "head": ((0, 0, 1.60), (0, 0, 1.86), "neck"),
    "upper_arm.L": ((0.175, 0, 1.49), (0.49, 0, 1.49), "chest"),
    "forearm.L": ((0.49, 0, 1.49), (0.73, 0, 1.49), "upper_arm.L"),
    "hand.L": ((0.73, 0, 1.49), (0.90, 0, 1.49), "forearm.L"),
    "upper_arm.R": ((-0.175, 0, 1.49), (-0.49, 0, 1.49), "chest"),
    "forearm.R": ((-0.49, 0, 1.49), (-0.73, 0, 1.49), "upper_arm.R"),
    "hand.R": ((-0.73, 0, 1.49), (-0.90, 0, 1.49), "forearm.R"),
    "thigh.L": ((0.112, 0, 0.98), (0.112, 0, 0.53), "spine"),
    "shin.L": ((0.112, 0, 0.53), (0.112, 0, 0.12), "thigh.L"),
    "foot.L": ((0.112, 0, 0.12), (0.112, 0.16, 0.03), "shin.L"),
    "thigh.R": ((-0.112, 0, 0.98), (-0.112, 0, 0.53), "spine"),
    "shin.R": ((-0.112, 0, 0.53), (-0.112, 0, 0.12), "thigh.R"),
    "foot.R": ((-0.112, 0, 0.12), (-0.112, 0.16, 0.03), "shin.R"),
}


def build_armature():
    ad = bpy.data.armatures.new("FighterArmature")
    arm = bpy.data.objects.new("FighterArmature", ad)
    bpy.context.collection.objects.link(arm)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode="EDIT")
    eb = {}
    for n, (h, t, p) in BONES.items():
        b = ad.edit_bones.new(n)
        b.head, b.tail = Vector(h), Vector(t)
        eb[n] = b
    for n, (h, t, p) in BONES.items():
        if p:
            eb[n].parent = eb[p]
    bpy.ops.object.mode_set(mode="OBJECT")
    return arm


def bind(obj, arm):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        print("[bind] bone-heat weights ok")
    except Exception as e:  # noqa: BLE001
        print(f"[bind] heat failed ({e}), rigid fallback")
        obj.parent = arm
        m = obj.modifiers.new("Bind", "ARMATURE")
        m.object = arm
        vg = obj.vertex_groups.new(name="spine")
        vg.add(list(range(len(obj.data.vertices))), 1.0, "REPLACE")


# ------------------------------------------------------------------ head
def build_head(skin_mat, lip_mat):
    parts = []
    h = ball("Head", (0, 0.004, 1.706), 0.118, skin_mat, "head",
             scale=(0.90, 0.98, 1.04), subsurf=2, seg=48, rings=32)
    sculpt(h, [
        ((0.000, 0.098, 1.780), (0.084, 0.020), 0.012, F),   # brow ridge
        ((0.045, 0.100, 1.740), (0.034, 0.019), -0.010, F),  # eye socket L
        ((-0.045, 0.100, 1.740), (0.034, 0.019), -0.010, F),
        ((0.076, 0.066, 1.706), (0.038, 0.034), 0.014, F),   # cheekbone L
        ((-0.076, 0.066, 1.706), (0.038, 0.034), 0.014, F),
        ((0.066, 0.080, 1.662), (0.028, 0.024), -0.008, F),  # hollow L
        ((-0.066, 0.080, 1.662), (0.028, 0.024), -0.008, F),
        ((0.000, 0.070, 1.640), (0.074, 0.036), 0.013, F),   # jaw
        ((0.000, 0.082, 1.610), (0.038, 0.024), 0.011, F),   # chin
        ((0.088, 0.000, 1.644), (0.026, 0.048), -0.018, L),
        ((-0.088, 0.000, 1.644), (0.026, 0.048), -0.018, R),
        ((0.000, -0.008, 1.822), (0.120, 0.056), 0.024, (0, 0, 1)),   # hair mass
        ((0.000, -0.086, 1.780), (0.110, 0.068), 0.016, B),
    ])
    parts.append(h)
    parts.append(ball("NoseBridge", (0, 0.110, 1.722), 0.013, skin_mat, "head",
                      scale=(0.85, 0.95, 1.40), subsurf=1))
    parts.append(ball("NoseTip", (0, 0.120, 1.684), 0.017, skin_mat, "head",
                      scale=(1.05, 0.92, 0.85), subsurf=1))
    parts.append(ball("NoseWing.L", (0.015, 0.113, 1.680), 0.011, skin_mat,
                      "head", scale=(1.0, 0.85, 0.8), subsurf=1))
    parts.append(ball("NoseWing.R", (-0.015, 0.113, 1.680), 0.011, skin_mat,
                      "head", scale=(1.0, 0.85, 0.8), subsurf=1))
    parts.append(ball("Ear.L", (0.098, -0.004, 1.706), 0.027, skin_mat, "head",
                      scale=(0.38, 0.72, 1.02), subsurf=1))
    parts.append(ball("Ear.R", (-0.098, -0.004, 1.706), 0.027, skin_mat, "head",
                      scale=(0.38, 0.72, 1.02), subsurf=1))
    return parts


def apply_face_texture(head, path, crop, bounds, min_front=0.05):
    me = head.data
    src = bpy.data.images.load(path)
    w, h = src.size
    px = np.empty(w * h * 4, dtype=np.float32)
    src.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)
    x0, y0, x1, y1 = [int(v) for v in crop]
    by0, by1 = h - y1, h - y0
    crop_px = px[by0:by1, x0:x1, :].copy()
    ch, cw = crop_px.shape[:2]
    skin = crop_px[ch // 2:int(ch * 0.8), cw // 4:int(cw * 0.75), :3].reshape(-1, 3).mean(axis=0)
    # flood the black backdrop in from the borders, then dilate to feather it
    dark = crop_px[:, :, :3].mean(axis=2) < 0.06
    m = np.zeros_like(dark)
    m[0], m[-1], m[:, 0], m[:, -1] = dark[0], dark[-1], dark[:, 0], dark[:, -1]
    for _ in range(cw + ch):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        g &= dark
        if np.array_equal(g, m):
            break
        m = g
    for _ in range(3):
        g = m.copy()
        g[1:] |= m[:-1]; g[:-1] |= m[1:]
        g[:, 1:] |= m[:, :-1]; g[:, :-1] |= m[:, 1:]
        m = g
    crop_px[m, :3] = skin
    pad = int(0.30 * max(ch, cw))
    nw, nh = cw + 2 * pad, ch + 2 * pad
    canvas = np.ones((nh, nw, 4), dtype=np.float32)
    canvas[:, :, :3] = skin
    canvas[pad:pad + ch, pad:pad + cw] = crop_px
    tex = bpy.data.images.new("FaceTex", nw, nh, alpha=True)
    tex.pixels.foreach_set(canvas.ravel())
    try:
        tex.pack()
    except Exception:  # noqa: BLE001
        pass
    fm = mat("FaceTex", (1, 1, 1), roughness=0.5)
    b = fm.node_tree.nodes.get("Principled BSDF")
    tn = fm.node_tree.nodes.new("ShaderNodeTexImage")
    tn.image = tex
    fm.node_tree.links.new(tn.outputs["Color"], b.inputs["Base Color"])
    me.materials.append(fm)
    fidx = len(me.materials) - 1
    while me.uv_layers:
        me.uv_layers.remove(me.uv_layers[0])
    uv = me.uv_layers.new(name="FaceUV")
    uv.data  # noqa: B018
    uvl = me.uv_layers.active.data
    xmin, xmax, zmin, zmax = bounds
    mw = head.matrix_world
    hair_z = zmin + (zmax - zmin) * 0.80
    hu0, hu1, hv = 0.34, 0.50, 0.92
    n = 0
    for p in me.polygons:
        c = mw @ p.center
        front = p.normal.y > min_front
        upper = c.z > hair_z
        if front or upper or p.normal.z > 0.45:
            p.material_index = fidx
            n += 1
        for li in p.loop_indices:
            v = mw @ me.vertices[me.loops[li].vertex_index].co
            if front or p.normal.z > 0.45 or upper:
                u01 = min(max((v.x - xmin) / (xmax - xmin), -0.2), 1.2)
                t01 = min(max((v.z - zmin) / (zmax - zmin), -0.2), 1.2)
                cx, cy = pad + u01 * cw, pad + t01 * ch
            else:
                cx = (hu0 + (hu1 - hu0) * (0.5 + 0.5 * v.x / max(xmax, 1e-6))) * nw
                cy = hv * nh
            uvl[li].uv = (cx / nw, cy / nh)
    me.update()
    print(f"[face] {n}/{len(me.polygons)} head polys textured")


# ------------------------------------------------------------------ extras
def build_extras(skin_mat, hair_mat, red, white, gold):
    parts = []
    # Muay Thai shorts: shaped cylinders hold form; spheres become diapers
    trunk = cyl("ShortsTrunk", (0, 0, 1.035), 0.155, 0.26, red, "spine",
                subsurf=2, verts=24)
    parts.append(shape_axis(trunk, [(0, 1.30), (0.5, 1.26), (1.0, 1.22)],
                            [(0, 1.02), (0.5, 1.0), (1.0, 0.96)]))
    for s, side in ((1, "L"), (-1, "R")):
        leg = cyl(f"ShortLeg.{side}", (s * 0.104, 0, 0.860), 0.132, 0.40,
                  red, f"thigh.{side}", subsurf=2, verts=24, cuts=3)
        parts.append(shape_axis(leg, [(0, 1.10), (0.4, 1.02), (1, 0.94)],
                                [(0, 1.10), (0.4, 1.02), (1, 0.94)]))
    parts.append(shape_axis(
        cyl("Waistband", (0, 0, 1.170), 0.157, 0.026, white, "spine",
            subsurf=2, verts=24),
        [(0, 1.26), (1.0, 1.24)], [(0, 1.02), (1.0, 1.0)]))
    # hand wraps + thumb + ankle wraps + chain
    for s, side in ((1, "L"), (-1, "R")):
        parts.append(ball(f"Thumb.{side}", (s * 0.792, 0.050, 1.500), 0.026,
                          skin_mat, f"hand.{side}", scale=(1.0, 1.20, 0.82),
                          subsurf=1))
        parts.append(shape_axis(
            cyl(f"WrapFist.{side}", (s * 0.812, 0, 1.490), 0.070, 0.10, red,
                f"hand.{side}", (0, math.radians(90), 0), subsurf=1, cuts=2),
            [(0, 0.94), (0.5, 1.0), (1, 0.94)], [(0, 0.94), (0.5, 1.0), (1, 0.94)]))
        parts.append(shape_axis(
            cyl(f"WrapWrist.{side}", (s * 0.700, 0, 1.490), 0.052, 0.10, red,
                f"forearm.{side}", (0, math.radians(90), 0), subsurf=1, cuts=3),
            [(0, 1.05), (1, 0.92)], [(0, 1.05), (1, 0.92)]))
        parts.append(shape_axis(
            cyl(f"AnkleWrap.{side}", (s * 0.112, 0, 0.135), 0.056, 0.09, red,
                f"shin.{side}", subsurf=1, cuts=2),
            [(0, 1.04), (1, 0.92)], [(0, 1.04), (1, 0.92)]))
    parts.append(torus("Chain", (0, 0, 1.545), 0.072, 0.008, gold, "chest"))
    return parts


def build_ink(parts, t=0.011):
    """Inverted hulls. Do NOT flip normals: three draws BackSide, and after a
    flip BackSide is the NEAR shell, which blacks the whole model out."""
    im = mat("Ink", (0.014, 0.014, 0.018), roughness=1.0)
    im.use_backface_culling = True
    arm = bpy.data.objects["FighterArmature"]
    hulls = []
    for o in parts:
        h = o.copy()
        h.data = o.data.copy()
        h.name = "Ink_" + o.name
        bpy.context.collection.objects.link(h)
        me = h.data
        n = len(me.vertices)
        if n:
            a = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get("co", a)
            a = a.reshape(-1, 3) + normals_of(me) * t
            me.vertices.foreach_set("co", a.ravel())
            me.update()
        for m in h.modifiers:
            if m.type == "SUBSURF":
                m.levels = m.render_levels = 1
        if not any(m.type == "ARMATURE" for m in h.modifiers):
            m = h.modifiers.new("Bind", "ARMATURE")
            m.object = arm
        # copy every weight so the hull deforms with the body
        for vg in o.vertex_groups:
            if vg.name not in h.vertex_groups:
                h.vertex_groups.new(name=vg.name)
        for v in o.data.vertices:
            for g in v.groups:
                h.vertex_groups[o.vertex_groups[g.group].name].add(
                    [v.index], g.weight, "REPLACE")
        h.data.materials.clear()
        h.data.materials.append(im)
        h.hide_render = True
        hulls.append(h)
    return hulls


# ------------------------------------------------------------------ main
def build(face=None, crop=(0, 0, 1, 1), bounds=(0, 1, 0, 1), no_ink=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    skin = mat("Skin", (0.78, 0.58, 0.47), roughness=0.55)
    hair = mat("Hair", (0.62, 0.55, 0.46), roughness=0.9)
    red = mat("ShortsRed", (0.62, 0.05, 0.08), roughness=0.65)
    white = mat("TrimWhite", (0.90, 0.90, 0.91), roughness=0.6)
    gold = mat("Gold", (0.80, 0.60, 0.24), roughness=0.35, metallic=0.8)
    lip = mat("Lips", (0.42, 0.19, 0.17), roughness=0.6)

    build_armature()
    arm = bpy.data.objects["FighterArmature"]
    body = build_body(skin)
    bind(body, arm)
    parts = [body] + build_head(skin, lip) + build_extras(
        skin, hair, red, white, gold)
    if face:
        # the photo carries the nose/eyes/brows/lips; drop the geometry
        for nm in ("NoseBridge", "NoseTip", "NoseWing.L", "NoseWing.R"):
            o = bpy.data.objects.get(nm)
            if o:
                parts.remove(o)
                bpy.data.objects.remove(o, do_unlink=True)
        apply_face_texture(bpy.data.objects["Head"], face, crop, bounds)
    for o in parts:
        try:
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:  # noqa: BLE001
            try:
                bpy.ops.object.mode_set(mode="OBJECT")
            except Exception:  # noqa: BLE001
                pass
    hulls = [] if no_ink else build_ink(parts)
    print(f"[build] {len(parts)} meshes + {len(hulls)} hulls, "
          f"{len(BONES)} bones")
    return parts, hulls


def parse(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--fbx", default=None)
    p.add_argument("--glb", default=None)
    p.add_argument("--face", default=None)
    p.add_argument("--face-crop", default="0,0,1,1")
    p.add_argument("--face-bounds", default="-0.1,0.1,1.61,1.86")
    p.add_argument("--no-ink", action="store_true")
    return p.parse_args(argv)


def main(argv):
    a = parse(argv)
    if bpy is None:
        raise RuntimeError("run with: blender --background --python "
                           "scripts/build_fighter.py")
    crop = tuple(float(x) for x in a.face_crop.split(","))
    bounds = tuple(float(x) for x in a.face_bounds.split(","))
    build(face=a.face, crop=crop, bounds=bounds, no_ink=a.no_ink)
    parts = [o for o in bpy.data.objects
             if o.type == "MESH" and not o.name.startswith("Ink_")]
    hulls = [o for o in bpy.data.objects
             if o.type == "MESH" and o.name.startswith("Ink_")]
    bpy.ops.wm.save_as_mainfile(filepath=a.out)
    print(f"[build] saved {a.out}")
    arm = bpy.data.objects["FighterArmature"]

    def sel(objs):
        bpy.ops.object.select_all(action="DESELECT")
        for o in objs:
            o.select_set(True)
        arm.select_set(True)

    if a.fbx:
        sel(parts)
        bpy.ops.export_scene.fbx(filepath=a.fbx, use_selection=True,
                                 add_leaf_bones=False, use_mesh_modifiers=True)
        print(f"[build] exported {a.fbx}")
    if a.glb:
        sel(parts + hulls)
        bpy.ops.export_scene.gltf(filepath=a.glb, export_format="GLB",
                                  use_selection=True, export_apply=True)
        print(f"[build] exported {a.glb}")


if __name__ == "__main__":
    main(sys.argv)
