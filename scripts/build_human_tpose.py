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


def _copy_vgroups(src, dst):
    for vg in src.vertex_groups:
        if vg.name not in dst.vertex_groups:
            dst.vertex_groups.new(name=vg.name)
    for v in src.data.vertices:
        for g in v.groups:
            name = src.vertex_groups[g.group].name
            dst.vertex_groups[name].add([v.index], g.weight, "REPLACE")


def build_ink(parts, thickness=0.011):
    """Inverted-hull ink outlines, SFIV style.

    Offset each hull along its own vertex normals (not object scale, which
    would vary with distance from the origin on a full body), flip the
    normals, and copy skin weights so hulls follow poses. The hull carries
    no slicer-facing role: excluded from the FBX export, included in the
    GLB where the viewer draws it BackSide.
    """
    import numpy as np
    ink_mat = mat("Ink", (0.015, 0.015, 0.02), roughness=1.0)
    ink_mat.use_backface_culling = True
    arm = bpy.data.objects["FighterArmature"]
    hulls = []
    bpy.ops.object.select_all(action="DESELECT")
    for o in parts:
        h = o.copy()
        h.data = o.data.copy()
        h.name = "Ink_" + o.name
        bpy.context.collection.objects.link(h)
        me = h.data
        n = len(me.vertices)
        if n:
            co = np.empty(n * 3, dtype=np.float64)
            me.vertices.foreach_get("co", co)
            nrm = _vertex_normals(me)
            co = co.reshape(-1, 3) + nrm * thickness
            me.vertices.foreach_set("co", co.ravel())
            me.update()
        for m in h.modifiers:
            if m.type == "SUBSURF":
                m.levels = 1
                m.render_levels = 1
        if not any(m.type == "ARMATURE" for m in h.modifiers):
            mod = h.modifiers.new("ArmatureBind", "ARMATURE")
            mod.object = arm
        _copy_vgroups(o, h)
        bpy.context.view_layer.objects.active = h
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.flip_normals()
        bpy.ops.object.mode_set(mode="OBJECT")
        h.data.materials.clear()
        h.data.materials.append(ink_mat)
        # stills use Freestyle lines instead (preview.py)
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
# Continuous body via the Skin modifier: a stick figure with per-vertex
# elliptical radii, skinned + subdivided into ONE manifold mesh. This is
# what kills the "turned on a lathe" look - real junctions instead of
# stacked primitives - and it still slices fine for the game because
# render_fighter_parts.py works off vertex groups, not separate objects.
BODY_SKELETON = [
    # (x, y, z, radius_x, radius_y) in metres, T-pose
    (0.000, 0.000, 0.985, 0.132, 0.098),   # 0 pelvis
    (0.000, 0.000, 1.150, 0.112, 0.090),   # 1 belly
    (0.000, 0.000, 1.330, 0.132, 0.097),   # 2 lower chest
    (0.000, 0.000, 1.455, 0.170, 0.105),   # 3 upper chest
    (0.000, 0.000, 1.560, 0.078, 0.072),   # 4 neck base
    (0.000, 0.000, 1.640, 0.064, 0.062),   # 5 neck top
    # left arm
    (0.180, 0.000, 1.495, 0.086, 0.086),   # 6 shoulder
    (0.262, 0.000, 1.490, 0.093, 0.089),   # 7 deltoid
    (0.365, 0.000, 1.490, 0.075, 0.071),   # 8 bicep
    (0.490, 0.000, 1.490, 0.057, 0.056),   # 9 elbow
    (0.600, 0.000, 1.490, 0.063, 0.058),   # 10 forearm
    (0.735, 0.000, 1.490, 0.044, 0.042),   # 11 wrist
    (0.845, 0.000, 1.490, 0.066, 0.059),   # 12 fist
    # right arm
    (-0.180, 0.000, 1.495, 0.086, 0.086),
    (-0.262, 0.000, 1.490, 0.093, 0.089),
    (-0.365, 0.000, 1.490, 0.075, 0.071),
    (-0.490, 0.000, 1.490, 0.057, 0.056),
    (-0.600, 0.000, 1.490, 0.063, 0.058),
    (-0.735, 0.000, 1.490, 0.044, 0.042),
    (-0.845, 0.000, 1.490, 0.066, 0.059),
    # left leg
    (0.115, 0.000, 0.985, 0.108, 0.104),
    (0.115, 0.000, 0.800, 0.108, 0.103),
    (0.115, 0.000, 0.610, 0.080, 0.078),
    (0.115, 0.000, 0.530, 0.072, 0.072),
    (0.115, 0.000, 0.380, 0.074, 0.070),
    (0.115, 0.000, 0.130, 0.050, 0.049),
    (0.115, 0.055, 0.062, 0.047, 0.066),   # foot
    # right leg
    (-0.115, 0.000, 0.985, 0.108, 0.104),
    (-0.115, 0.000, 0.800, 0.108, 0.103),
    (-0.115, 0.000, 0.610, 0.080, 0.078),
    (-0.115, 0.000, 0.530, 0.072, 0.072),
    (-0.115, 0.000, 0.380, 0.074, 0.070),
    (-0.115, 0.000, 0.130, 0.050, 0.049),
    (-0.115, 0.055, 0.062, 0.047, 0.066),
]
BODY_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (3, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (3, 13), (13, 14), (14, 15), (15, 16), (16, 17), (17, 18), (18, 19),
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
    (0, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33),
]


def build_continuous_body(S, material):
    me = bpy.data.meshes.new("Body")
    obj = bpy.data.objects.new("Body", me)
    bpy.context.collection.objects.link(obj)
    import bmesh
    bm = bmesh.new()
    vs = [bm.verts.new((v[0], v[1], v[2])) for v in BODY_SKELETON]
    bm.verts.ensure_lookup_table()
    for a, b in BODY_EDGES:
        bm.edges.new((vs[a], vs[b]))
    bm.to_mesh(me)
    bm.free()
    obj.modifiers.new("Skin", "SKIN")
    for i, v in enumerate(BODY_SKELETON):
        sv = me.skin_vertices[0].data[i]
        sv.radius = (v[3], v[4])
    me.skin_vertices[0].data[0].use_root = True
    sub = obj.modifiers.new("Sub", "SUBSURF")
    sub.levels = 4
    sub.render_levels = 4
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier="Skin")
    bpy.ops.object.modifier_apply(modifier="Sub")
    for p in obj.data.polygons:
        p.use_smooth = True
    obj.data.materials.append(material)
    sculpt_normals(obj, MUSCLE_BUMPS)
    return obj


def bind_body(obj, arm):
    """Skin the continuous body to the armature (bone-heat weights, with a
    distance fallback)."""
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.parent_set(type="ARMATURE_AUTO")
        print("[build] body bound with bone-heat weights")
    except Exception as e:  # noqa: BLE001
        print(f"[build] auto weights failed ({e}); falling back")
        obj.parent = arm
        mod = obj.modifiers.new("ArmatureBind", "ARMATURE")
        mod.object = arm
        _distance_weights(obj, arm)


def _distance_weights(obj, arm, k=2, power=4.0):
    from mathutils import Vector

    def seg_dist(p, a, b):
        ab = b - a
        t = max(0.0, min(1.0, (p - a).dot(ab) / max(ab.length_squared, 1e-9)))
        return (p - (a + ab * t)).length

    bones = [(b.name, Vector(b.head_local), Vector(b.tail_local))
             for b in arm.data.bones]
    groups = {name: obj.vertex_groups.new(name=name) for name, _, _ in bones}
    for v in obj.data.vertices:
        p = Vector(v.co)
        ds = sorted(((seg_dist(p, h, t), n) for n, h, t in bones))[:k]
        ws = [(n, 1.0 / max(d, 1e-4) ** power) for d, n in ds]
        tot = sum(w for _, w in ws)
        for n, w in ws:
            groups[n].add([v.index], w / tot, "REPLACE")


def _vertex_normals(me):
    """Per-vertex normals averaged from polygons.

    mesh.vertex_normals.foreach_get() is unreliable on a mesh that was
    just written by modifier_apply, so compute them directly.
    """
    import numpy as np
    n = len(me.vertices)
    acc = np.zeros((n, 3), dtype=np.float64)
    pn = np.empty(len(me.polygons) * 3, dtype=np.float64)
    me.polygons.foreach_get("normal", pn)
    pn = pn.reshape(-1, 3)
    for i, p in enumerate(me.polygons):
        nv = pn[i]
        for v in p.vertices:
            acc[v] += nv
    ln = np.linalg.norm(acc, axis=1, keepdims=True)
    ln[ln == 0] = 1.0
    return acc / ln


def sculpt_normals(obj, bumps):
    """Muscle bulges that push along the surface normal.

    bumps: (center, (sigma_x, sigma_z), amp, direction). A vertex is
    affected if it lies in the muscle's x/z region AND faces the given
    direction, then moves out along its own normal. This is what makes a
    rounded belly of muscle instead of a flat slab - the failure mode of
    a plain y-gaussian on a curved surface.
    """
    import numpy as np
    me = obj.data
    mw = np.array(obj.matrix_world)
    n = len(me.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    nr = _vertex_normals(me)
    wco = co @ mw[:3, :3].T + mw[:3, 3]
    nrm_mat = np.linalg.inv(mw[:3, :3]).T
    wn = nr @ nrm_mat.T
    wn /= np.maximum(np.linalg.norm(wn, axis=1, keepdims=True), 1e-9)
    inv = np.linalg.inv(mw[:3, :3])
    disp = np.zeros_like(co)
    for center, sigma, amp, direction in bumps:
        d = np.array(direction, float)
        d /= np.linalg.norm(d)
        w = np.exp(-(((wco[:, 0] - center[0]) / sigma[0]) ** 2
                     + ((wco[:, 2] - center[2]) / sigma[1]) ** 2))
        face = np.clip(wn @ d, 0.0, 1.0)
        w = w * face
        disp += (wn * (w * amp)[:, None]) @ inv.T
    mag = np.linalg.norm(disp, axis=1)
    print(f"[sculpt] {obj.name}: verts={n} max_disp={mag.max():.4f} "
          f"mean_disp={mag.mean():.4f} normal_ok={np.isfinite(wn).all()}")
    me.vertices.foreach_set("co", (co + disp).ravel())
    me.update()
    return obj


front_v = (0, 1, 0)
back_v = (0, -1, 0)
left_v = (1, 0, 0)
right_v = (-1, 0, 0)

# (center, (sigma_x, sigma_z), amplitude, facing direction)
MUSCLE_BUMPS = [
    # ---- chest ----
    ((0.098, 0, 1.452), (0.080, 0.060), 0.072, front_v),      # pec L
    ((-0.098, 0, 1.452), (0.080, 0.060), 0.072, front_v),     # pec R
    ((0.000, 0, 1.448), (0.014, 0.078), -0.052, front_v),     # sternum
    # ---- abs: 3 rows, with grooves between ----
    ((0.050, 0, 1.322), (0.032, 0.030), 0.056, front_v),
    ((-0.050, 0, 1.322), (0.032, 0.030), 0.056, front_v),
    ((0.050, 0, 1.242), (0.032, 0.030), 0.053, front_v),
    ((-0.050, 0, 1.242), (0.032, 0.030), 0.053, front_v),
    ((0.050, 0, 1.165), (0.032, 0.029), 0.045, front_v),
    ((-0.050, 0, 1.165), (0.032, 0.029), 0.045, front_v),
    ((0.000, 0, 1.245), (0.010, 0.100), -0.042, front_v),     # linea alba
    ((0.000, 0, 1.283), (0.062, 0.011), -0.040, front_v),     # ab groove
    ((0.000, 0, 1.203), (0.062, 0.011), -0.040, front_v),     # ab groove
    # ---- obliques / serratus ----
    ((0.126, 0.015, 1.230), (0.026, 0.075), 0.028, front_v),
    ((-0.126, 0.015, 1.230), (0.026, 0.075), 0.028, front_v),
    # ---- lats: widen the upper torso ----
    ((0.176, 0, 1.360), (0.050, 0.135), 0.062, left_v),
    ((-0.176, 0, 1.360), (0.050, 0.135), 0.062, right_v),
    # ---- back ----
    ((0.000, 0, 1.330), (0.014, 0.190), -0.030, back_v),      # spine groove
    ((0.086, 0, 1.518), (0.056, 0.054), 0.050, back_v),       # trap L
    ((-0.086, 0, 1.518), (0.056, 0.054), 0.050, back_v),      # trap R
    ((0.070, 0, 1.330), (0.045, 0.090), 0.030, back_v),       # lat back L
    ((-0.070, 0, 1.330), (0.045, 0.090), 0.030, back_v),      # lat back R
    ((0.105, 0, 0.975), (0.072, 0.080), 0.042, back_v),       # glute L
    ((-0.105, 0, 0.975), (0.072, 0.080), 0.042, back_v),      # glute R
    # ---- arms ----
    ((0.272, 0, 1.492), (0.062, 0.072), 0.060, left_v),       # delt L (out)
    ((-0.272, 0, 1.492), (0.062, 0.072), 0.060, right_v),     # delt R
    ((0.272, 0, 1.510), (0.060, 0.055), 0.026, back_v),       # rear delt L
    ((-0.272, 0, 1.510), (0.060, 0.055), 0.026, back_v),
    ((0.372, 0, 1.492), (0.082, 0.050), 0.064, front_v),      # bicep L
    ((-0.372, 0, 1.492), (0.082, 0.050), 0.064, front_v),     # bicep R
    ((0.382, 0, 1.492), (0.086, 0.054), 0.046, back_v),       # tricep L
    ((-0.382, 0, 1.492), (0.086, 0.054), 0.046, back_v),      # tricep R
    ((0.600, 0, 1.492), (0.050, 0.046), 0.024, front_v),      # brachio L
    ((-0.600, 0, 1.492), (0.050, 0.046), 0.024, front_v),
    # ---- legs ----
    ((0.115, 0, 0.805), (0.084, 0.112), 0.068, front_v),      # quad L
    ((-0.115, 0, 0.805), (0.084, 0.112), 0.068, front_v),     # quad R
    ((0.115, 0, 0.820), (0.078, 0.104), 0.046, back_v),       # ham L
    ((-0.115, 0, 0.820), (0.078, 0.104), 0.046, back_v),      # ham R
    ((0.118, 0, 0.392), (0.056, 0.086), 0.064, back_v),       # calf L
    ((-0.118, 0, 0.392), (0.056, 0.086), 0.064, back_v),      # calf R
    ((0.115, 0, 0.300), (0.050, 0.095), 0.018, front_v),      # shin L
    ((-0.115, 0, 0.300), (0.050, 0.095), 0.018, front_v),     # shin R
]


def build_head(S, H, HD, LIPS, DARK):
    """Detailed head: skull sculpted with jaw/brow/cheek/nose mass, then
    small feature parts. Overlaps the neck so there is no seam."""
    parts = []
    head = ball("Head", (0, 0.008, 1.735), 0.120, S, "head",
                scale=(0.94, 1.02, 1.04), subsurf=2, seg=32, rings=24)
    sculpt(head, [
        # flatten the face plane, keep the back round
        ((0.000, 0.090, 1.720), (0.090, 0.060, 0.070), -0.016, (0, 1, 0), None),
        # brow ridge
        ((0.000, 0.095, 1.790), (0.085, 0.060, 0.022), 0.010, (0, 1, 0), None),
        # cheekbones
        ((0.072, 0.070, 1.720), (0.045, 0.060, 0.045), 0.012, (0, 1, 0), None),
        ((-0.072, 0.070, 1.720), (0.045, 0.060, 0.045), 0.012, (0, 1, 0), None),
        # jaw + chin
        ((0.000, 0.060, 1.650), (0.075, 0.060, 0.045), 0.014, (0, 1, 0), None),
        ((0.000, 0.070, 1.628), (0.040, 0.060, 0.030), 0.010, (0, 1, 0), None),
        # narrow the jaw at the sides
        ((0.086, 0.000, 1.650), (0.030, 0.080, 0.050), -0.016, (1, 0, 0), (0, 1)),
        ((-0.086, 0.000, 1.650), (0.030, 0.080, 0.050), -0.016, (-1, 0, 0), (0, -1)),
        # back of skull tuck
        ((0.000, -0.095, 1.760), (0.070, 0.050, 0.070), -0.012, (0, -1, 0), None),
    ])
    parts.append(head)
    parts.append(ball("NoseBridge", (0, 0.108, 1.740), 0.017, S, "head",
                      scale=(0.80, 0.95, 1.55), subsurf=1))
    parts.append(ball("NoseTip", (0, 0.122, 1.706), 0.021, S, "head",
                      scale=(1.05, 0.95, 0.85), subsurf=1))
    parts.append(ball("NoseWing.L", (0.019, 0.114, 1.704), 0.014, S, "head",
                      scale=(1.0, 0.85, 0.85), subsurf=1))
    parts.append(ball("NoseWing.R", (-0.019, 0.114, 1.704), 0.014, S, "head",
                      scale=(1.0, 0.85, 0.85), subsurf=1))
    parts.append(ball("Ear.L", (0.105, -0.004, 1.740), 0.030, S, "head",
                      scale=(0.42, 0.78, 1.08), subsurf=1))
    parts.append(ball("Ear.R", (-0.105, -0.004, 1.740), 0.030, S, "head",
                      scale=(0.42, 0.78, 1.08), subsurf=1))
    # almond eyes, straight dark brows
    parts.append(ball("Eye.L", (0.046, 0.110, 1.768), 0.017, DARK, "head",
                      scale=(1.25, 0.78, 0.54)))
    parts.append(ball("Eye.R", (-0.046, 0.110, 1.768), 0.017, DARK, "head",
                      scale=(1.25, 0.78, 0.54)))
    parts.append(cube("Brow.L", (0.055, 0.112, 1.798), (0.027, 0.008, 0.005),
                      DARK, "head", subsurf=1))
    parts.append(cube("Brow.R", (-0.055, 0.112, 1.798), (0.027, 0.008, 0.005),
                      DARK, "head", subsurf=1))
    parts.append(ball("LipUpper", (0, 0.118, 1.694), 0.026, LIPS, "head",
                      scale=(1.05, 0.30, 0.24), subsurf=1))
    parts.append(ball("LipLower", (0, 0.116, 1.687), 0.021, LIPS, "head",
                      scale=(0.95, 0.30, 0.28), subsurf=1))
    # hair: bleached, medium length, swept up and back
    parts.append(ball("Hair", (0, -0.018, 1.862), 0.126, H, "head",
                      scale=(1.02, 1.03, 0.58), subsurf=2))
    parts.append(ball("HairFront", (0, 0.052, 1.888), 0.082, H, "head",
                      scale=(1.10, 0.90, 0.42), subsurf=2))
    return parts


def build_body(S, H, HD, LIPS, SHADE, RED, WHITE, GOLD, DARK):
    parts = []
    # neck filler so the head sits on a column (skin stops at the neck)
    neck = cyl("Neck", (0, 0, 1.600), 0.070, 0.14, S, "neck", subsurf=2,
               cuts=3)
    parts.append(profile(neck, [(0, 1.10), (0.5, 1.02), (1, 0.95)]))
    parts += build_head(S, H, HD, LIPS, DARK)
    parts.append(torus("Chain", (0, 0, 1.560), 0.090, 0.008, GOLD, "chest"))
    # wrist wraps + fist wraps ride on the continuous arms
    for side, s in (("L", 1), ("R", -1)):
        wf = cyl(f"WrapFist.{side}", (s * 0.845, 0, 1.490), 0.077, 0.10, RED,
                 f"hand.{side}", (0, math.radians(90), 0), subsurf=1, cuts=2)
        parts.append(profile(wf, [(0, 0.94), (0.5, 1.0), (1, 0.94)]))
        ww = cyl(f"WrapWrist.{side}", (s * 0.740, 0, 1.490), 0.055, 0.10, RED,
                 f"forearm.{side}", (0, math.radians(90), 0), subsurf=1,
                 cuts=3)
        parts.append(profile(ww, [(0, 1.05), (1, 0.92)]))
        aw = cyl(f"AnkleWrap.{side}", (s * 0.115, 0, 0.150), 0.061, 0.09,
                 RED, f"shin.{side}", subsurf=1, cuts=2)
        parts.append(profile(aw, [(0, 1.04), (1, 0.90)]))
        parts.append(ball(f"Thumb.{side}", (s * 0.830, 0.058, 1.500), 0.029,
                          S, f"hand.{side}", scale=(1.0, 1.25, 0.82),
                          subsurf=1))
    # Muay Thai shorts: need to clear the glute bulge (y ~ -0.18)
    trunk = ball("ShortsTrunk", (0, 0, 1.030), 0.252, RED, "spine",
                 scale=(1.0, 0.94, 0.74), subsurf=2)
    parts.append(shape_axis(trunk, [(0, 1.0), (0.5, 1.0), (1.0, 0.90)],
                            [(0, 1.04), (0.5, 1.0), (1.0, 0.88)]))
    for side, s in (("L", 1), ("R", -1)):
        leg = cyl(f"ShortLeg.{side}", (s * 0.124, 0, 0.885), 0.142, 0.32,
                  RED, f"thigh.{side}", subsurf=2, cuts=4)
        parts.append(profile(leg, [(0, 1.14), (0.45, 1.02), (1, 0.86)]))
    band = ball("Waistband", (0, 0, 1.192), 0.226, WHITE, "spine",
                scale=(1.0, 0.78, 0.072), subsurf=2)
    parts.append(band)
    return parts


def build(do_dirt=False, skip_ink=False):
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
    arm = bpy.data.objects["FighterArmature"]
    # one continuous body mesh (organic junctions, muscle sculpted in)
    body = build_continuous_body(S, S)
    bind_body(body, arm)
    # accessories: head + features, wraps, shorts (rigid-bound)
    parts = [body] + build_body(S, H, HD, LIPS, SHADE, RED, WHITE, GOLD, DARK)
    if do_dirt:
        # NOTE: bpy's vertex_color_dirt bakes very dark values on these
        # smooth meshes (~0.2 mean). glTF multiplies COLOR_0 into base
        # color, so exporting it makes the whole model look unlit/dark.
        # Off by default; only enable if the values are checked first.
        apply_dirt(parts)
    hulls = [] if skip_ink else build_ink(parts)
    print(f"[build] fighter done: {len(parts)} meshes"
          f" + {len(hulls)} ink hulls + armature, {len(BONES)} bones")
    return parts, hulls


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--fbx", default=None)
    p.add_argument("--glb", default=None)
    p.add_argument("--dirt", action="store_true",
                   help="bake crevice shading (darkens! check values first)")
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
    build(do_dirt=args.dirt, skip_ink=args.skip_ink)
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
        # do NOT force vertex colors - a dark baked attribute multiplies
        # into base color and makes the whole model look unlit.
        select_for_export(parts, hulls, with_ink=True)
        bpy.ops.export_scene.gltf(filepath=args.glb, export_format="GLB",
                                  use_selection=True)
        print(f"[build] exported {args.glb}")


if __name__ == "__main__":
    main(sys.argv)
