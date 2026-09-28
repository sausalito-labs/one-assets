"""Extract evaluated fighter triangles (with UVs + textures) to an .npz.

Runs inside Blender. Rasterizing lives in rasterize.py (PIL/numpy), because
Cycles/Freestyle/EEVEE are unusably slow or unavailable on this box.

    blender --background --python scripts/softrender.py -- \
        --blend characters/human_tpose/human_tpose.blend \
        --npz /tmp/fighter.npz --subsurf 1
"""
import argparse
import sys

import numpy as np
import bpy


def _material_image(mat):
    """Return (name, pixel_array) if the material has an image texture."""
    if not mat or not mat.use_nodes:
        return None
    tex = None
    for n in mat.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            tex = n
            break
    if tex is None:
        return None
    img = tex.image
    w, h = img.size
    px = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)[::-1, :, :3]  # bottom-up -> top-down RGB
    return img.name, np.clip(px, 0, 1)


def gather(blend, skip_ink=True, subsurf=None):
    bpy.ops.wm.open_mainfile(filepath=blend)
    if subsurf is not None:
        for o in bpy.data.objects:
            for m in o.modifiers:
                if m.type == "SUBSURF":
                    m.levels = subsurf
                    m.render_levels = subsurf
    dg = bpy.context.evaluated_depsgraph_get()
    tris, cols, uvs, texids = [], [], [], []
    images = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        if skip_ink and obj.name.startswith("Ink_"):
            continue
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        if len(me.vertices) == 0:
            ev.to_mesh_clear()
            continue
        mw = obj.matrix_world
        co = np.empty(len(me.vertices) * 3, dtype=np.float64)
        me.vertices.foreach_get("co", co)
        world = co.reshape(-1, 3) @ np.array(mw.to_3x3()).T \
            + np.array(mw.translation)
        uv_layer = me.uv_layers.active.data if me.uv_layers else None
        # material -> (color, texid)
        mslot = []
        for slot in obj.material_slots:
            m = slot.material
            col = (0.8, 0.8, 0.8)
            tid = -1
            if m:
                bsdf = m.node_tree.nodes.get("Principled BSDF") if m.use_nodes else None
                if bsdf:
                    col = tuple(bsdf.inputs["Base Color"].default_value[:3])
                got = _material_image(m)
                if got:
                    name, arr = got
                    if name not in images:
                        images[name] = (len(images), arr)
                    tid = images[name][0]
            mslot.append((col, tid))
        poly_tid = np.zeros(len(me.polygons), dtype=np.int32)
        if uv_layer is not None:
            uvarr = np.empty(len(me.loops) * 2, dtype=np.float64)
            uv_layer.foreach_get("uv", uvarr)
            uvarr = uvarr.reshape(-1, 2)
        else:
            uvarr = np.zeros((len(me.loops), 2))
        for i, p in enumerate(me.polygons):
            col, tid = mslot[p.material_index] if mslot else ((0.8, 0.8, 0.8), -1)
            vids = list(p.vertices)
            lids = list(p.loop_indices)
            for k in range(1, len(vids) - 1):
                tri = world[[vids[0], vids[k], vids[k + 1]]]
                tris.append(tri)
                cols.append(col)
                texids.append(tid)
                uvs.append([uvarr[lids[0]], uvarr[lids[k]],
                            uvarr[lids[k + 1]]])
        ev.to_mesh_clear()
    out = {
        "tris": np.array(tris, np.float32),
        "cols": np.array(cols, np.float32),
        "uvs": np.array(uvs, np.float32),
        "texids": np.array(texids, np.int32),
    }
    for name, (idx, arr) in images.items():
        out[f"img_{idx}"] = arr.astype(np.float32)
        out[f"imgname_{idx}"] = np.array([name])
    return out


def parse(argv):
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--subsurf", type=int, default=None)
    return ap.parse_args(argv)


if __name__ == "__main__":
    a = parse(sys.argv)
    data = gather(a.blend, subsurf=a.subsurf)
    np.savez_compressed(a.npz, **data)
    print(f"[extract] {len(data['tris'])} triangles -> {a.npz}")
