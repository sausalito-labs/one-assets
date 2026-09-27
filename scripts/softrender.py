"""Extract evaluated fighter triangles to an .npz for fast previewing.

Runs inside Blender (needs bpy for mesh evaluation + subsurf). The actual
rasterizing lives in rasterize.py (PIL, C-speed polygon fill), because
Cycles/Freestyle/EEVEE are unusable and a numpy per-pixel loop is too slow
on this 4-core box.

    blender --background --python scripts/softrender.py -- \
        --blend characters/human_tpose/human_tpose.blend \
        --npz /tmp/fighter.npz --subsurf 1
"""
import argparse
import sys

import numpy as np
import bpy


def gather(blend, skip_ink=True, subsurf=None):
    bpy.ops.wm.open_mainfile(filepath=blend)
    if subsurf is not None:
        for o in bpy.data.objects:
            for m in o.modifiers:
                if m.type == "SUBSURF":
                    m.levels = subsurf
                    m.render_levels = subsurf
    dg = bpy.context.evaluated_depsgraph_get()
    tris, cols = [], []
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
        co = co.reshape(-1, 3)
        world = co @ np.array(mw.to_3x3()).T + np.array(mw.translation)
        col = (0.8, 0.8, 0.8)
        if obj.material_slots and obj.material_slots[0].material:
            bsdf = obj.material_slots[0].material.node_tree.nodes.get(
                "Principled BSDF")
            if bsdf:
                col = tuple(bsdf.inputs["Base Color"].default_value[:3])
        for p in me.polygons:
            vids = list(p.vertices)
            for k in range(1, len(vids) - 1):
                tris.append(world[[vids[0], vids[k], vids[k + 1]]])
                cols.append(col)
        ev.to_mesh_clear()
    return np.array(tris, np.float32), np.array(cols, np.float32)


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
    tris, cols = gather(a.blend, subsurf=a.subsurf)
    np.savez_compressed(a.npz, tris=tris, cols=cols)
    print(f"[extract] {len(tris)} triangles -> {a.npz}")
