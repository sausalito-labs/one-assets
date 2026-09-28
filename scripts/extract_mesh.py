"""Extract evaluated triangles (+ UVs, textures, ink flags) to an .npz.

Runs inside Blender. Cycles/EEVEE/Freestyle are unusable headless on this
box (4 cores, no GPU), so rasterising is done by rasterize.py.

    blender --background --python scripts/extract_mesh.py -- \
        --blend characters/takeru/takeru.blend --npz /tmp/f.npz --subsurf 1
"""
import argparse
import sys

import numpy as np
import bpy


def _img_of(m):
    if not m or not m.use_nodes:
        return None
    for n in m.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image:
            im = n.image
            w, h = im.size
            a = np.empty(w * h * 4, dtype=np.float32)
            im.pixels.foreach_get(a)
            return im.name, np.clip(a.reshape(h, w, 4)[::-1, :, :3], 0, 1)
    return None


def gather(blend, subsurf=None, with_ink=False):
    bpy.ops.wm.open_mainfile(filepath=blend)
    if subsurf is not None:
        for o in bpy.data.objects:
            for m in o.modifiers:
                if m.type == "SUBSURF":
                    m.levels = m.render_levels = subsurf
    dg = bpy.context.evaluated_depsgraph_get()
    tris, cols, uvs, texids, isink = [], [], [], [], []
    imgs = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        ink = obj.name.startswith("Ink_")
        if ink and not with_ink:
            continue
        if obj.hide_render and not ink:
            continue
        ev = obj.evaluated_get(dg)
        me = ev.to_mesh()
        if not len(me.vertices):
            ev.to_mesh_clear()
            continue
        mw = obj.matrix_world
        co = np.empty(len(me.vertices) * 3)
        me.vertices.foreach_get("co", co)
        world = co.reshape(-1, 3) @ np.array(mw.to_3x3()).T + np.array(mw.translation)
        if me.uv_layers:
            ua = np.empty(len(me.loops) * 2)
            me.uv_layers.active.data.foreach_get("uv", ua)
            ua = ua.reshape(-1, 2)
        else:
            ua = np.zeros((len(me.loops), 2))
        slots = []
        for sl in obj.material_slots:
            m = sl.material
            col, tid = (0.8, 0.8, 0.8), -1
            if m and m.use_nodes:
                b = m.node_tree.nodes.get("Principled BSDF")
                if b:
                    col = tuple(b.inputs["Base Color"].default_value[:3])
                got = _img_of(m)
                if got:
                    nm, arr = got
                    if nm not in imgs:
                        imgs[nm] = (len(imgs), arr)
                    tid = imgs[nm][0]
            slots.append((col, tid))
        for p in me.polygons:
            col, tid = slots[p.material_index] if slots else ((0.8,) * 3, -1)
            vids = list(p.vertices)
            lids = list(p.loop_indices)
            for k in range(1, len(vids) - 1):
                tris.append(world[[vids[0], vids[k], vids[k + 1]]])
                cols.append(col)
                texids.append(tid)
                isink.append(1 if ink else 0)
                uvs.append([ua[lids[0]], ua[lids[k]], ua[lids[k + 1]]])
        ev.to_mesh_clear()
    out = {
        "tris": np.array(tris, np.float32),
        "cols": np.array(cols, np.float32),
        "uvs": np.array(uvs, np.float32),
        "texids": np.array(texids, np.int32),
        "isink": np.array(isink, np.int32),
    }
    for nm, (i, arr) in imgs.items():
        out[f"img_{i}"] = arr.astype(np.float32)
    return out


def main():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--subsurf", type=int, default=None)
    ap.add_argument("--with-ink", action="store_true")
    a = ap.parse_args(argv)
    d = gather(a.blend, a.subsurf, a.with_ink)
    np.savez_compressed(a.npz, **d)
    print(f"[extract] {len(d['tris'])} tris -> {a.npz}")


if __name__ == "__main__":
    main()
