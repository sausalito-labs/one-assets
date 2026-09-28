"""Rasterize extracted fighter triangles with PIL (fast, no GPU).

    python3 scripts/rasterize.py --npz /tmp/fighter.npz --out /tmp/p.png \
        --view threeq --size 560

Painter's algorithm (far-to-near) with flat Lambert shading, drawn at 2x
then downsampled for cheap antialiasing. Good enough to judge silhouette,
proportions and muscle; not a beauty render.
"""
import argparse
import math

import numpy as np
from PIL import Image, ImageDraw

VIEWS = {
    "front": ((0.0, 3.2, 1.20), (0, 0, 1.02), 40),
    "threeq": ((1.9, 2.7, 1.40), (0, 0, 1.02), 40),
    "side": ((3.2, 0.0, 1.20), (0, 0, 1.02), 40),
    "back": ((0.0, -3.2, 1.25), (0, 0, 1.06), 40),
    "head": ((0.30, 1.15, 1.72), (0, 0.02, 1.69), 36),
    "face": ((0.12, 0.62, 1.760), (0, 0.04, 1.745), 34),
    "head3q": ((0.95, 1.00, 1.74), (0, 0.02, 1.69), 36),
    "legs": ((0.0, 2.4, 0.65), (0, 0, 0.60), 40),
}


def basis(pos, target, fov, W, H, up=(0, 0, 1.0)):
    pos = np.array(pos, float)
    fwd = np.array(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    upv = np.array(up, float)
    right = np.cross(fwd, upv)
    right /= np.linalg.norm(right)
    tup = np.cross(right, fwd)
    f = (H / 2.0) / math.tan(math.radians(fov) / 2.0)
    return pos, right, tup, fwd, f


def zrender(proj, z, cols, shade, W, H, bg, keep, uvs=None, texids=None,
            images=None):
    """Accurate per-pixel z-buffer with optional texture sampling."""
    zbuf = np.full((H, W), 1e9)
    cbuf = np.zeros((H, W, 3))
    cbuf[:, :] = bg
    order = np.argsort(-z.min(axis=1))
    for i in order:
        if not keep[i]:
            continue
        p = proj[i]
        zz = z[i]
        x0 = max(int(p[:, 0].min()), 0)
        x1 = min(int(np.ceil(p[:, 0].max())), W - 1)
        y0 = max(int(p[:, 1].min()), 0)
        y1 = min(int(np.ceil(p[:, 1].max())), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        d = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
             - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(d) < 1e-9:
            continue
        w0 = ((p[1, 0] - gx) * (p[2, 1] - gy)
              - (p[2, 0] - gx) * (p[1, 1] - gy)) / d
        w1 = ((p[2, 0] - gx) * (p[0, 1] - gy)
              - (p[0, 0] - gx) * (p[2, 1] - gy)) / d
        w2 = 1.0 - w0 - w1
        mask = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        depth = w0 * zz[0] + w1 * zz[1] + w2 * zz[2]
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = mask & (depth < sub)
        if not hit.any():
            continue
        sub[hit] = depth[hit]
        base = np.array(cols[i], dtype=np.float64)
        tid = int(texids[i]) if texids is not None else -1
        if tid >= 0 and tid in images:
            img = images[tid]
            ih, iw = img.shape[:2]
            uv = uvs[i]
            U = w0 * uv[0, 0] + w1 * uv[1, 0] + w2 * uv[2, 0]
            V = w0 * uv[0, 1] + w1 * uv[1, 1] + w2 * uv[2, 1]
            ix = np.clip((U * iw).astype(np.int32), 0, iw - 1)
            iy = np.clip(((1.0 - V) * ih).astype(np.int32), 0, ih - 1)
            tex = img[iy, ix]
            c = tex * shade[i] * 255.0
        else:
            c = base.reshape(1, 1, 3) * shade[i] * 255.0
            c = np.broadcast_to(c, (y1 - y0 + 1, x1 - x0 + 1, 3))
        cbuf[y0:y1 + 1, x0:x1 + 1][hit] = np.clip(c[hit], 0, 255)
    return cbuf.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="threeq", choices=sorted(VIEWS))
    ap.add_argument("--size", type=int, default=560)
    ap.add_argument("--ss", type=int, default=2, help="supersample factor")
    ap.add_argument("--ambient", type=float, default=0.34)
    ap.add_argument("--browser", action="store_true",
                    help="simulate three.js: meshes FrontSide, Ink BackSide")
    ap.add_argument("--only-tex", action="store_true",
                    help="render only textured triangles (debug)")
    ap.add_argument("--debug-tex", action="store_true",
                    help="paint textured tris green, ignore the image")
    ap.add_argument("--cull", default="none",
                    choices=["none", "front", "back"],
                    help="cull faces by projected winding (simulate GL sides)")
    ap.add_argument("--z", action="store_true",
                    help="accurate z-buffer (slow) instead of painter's")
    a = ap.parse_args()

    d = np.load(a.npz)
    tris = d["tris"].astype(np.float64)
    cols = d["cols"].astype(np.float64)
    if a.only_tex and "texids" in d:
        m = d["texids"] >= 0
        orig = d
        d = {k: (orig[k][m] if (hasattr(orig[k], "shape")
                                and orig[k].shape[:1] == m.shape) else orig[k])
             for k in orig.files}
        tris = d["tris"].astype(np.float64)
        cols = d["cols"].astype(np.float64)
    uvs = d["uvs"].astype(np.float64) if "uvs" in d else None
    texids = d["texids"] if "texids" in d else None
    isink = d["isink"] if "isink" in d.keys() else None
    images = {int(k.split("_")[1]): d[k].astype(np.float64)
              for k in d.keys()
              if k.startswith("img_") and not k.startswith("imgname_")}
    W = H = a.size * a.ss
    pos, right, up, fwd, f = basis(*VIEWS[a.view], W, H)

    v = tris
    rel = v.reshape(-1, 3) - pos
    x = rel @ right
    y = rel @ up
    z = rel @ fwd
    sx = W / 2.0 + f * x / np.maximum(z, 1e-6)
    sy = H / 2.0 - f * y / np.maximum(z, 1e-6)
    proj = np.stack([sx, sy], axis=1).reshape(-1, 3, 2)
    z = z.reshape(-1, 3)

    # face normals + two-light flat shading
    fn = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    tocam = pos - v[:, 0]
    tocam /= np.maximum(np.linalg.norm(tocam, axis=1, keepdims=True), 1e-9)
    flip = (fn * tocam).sum(axis=1) < 0
    fn[flip] *= -1
    L1 = np.array([0.40, 0.72, 0.57]); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.62, 0.30, 0.24]); L2 /= np.linalg.norm(L2)
    lam1 = np.clip(fn @ L1, 0, 1)
    lam2 = np.clip(fn @ L2, 0, 1)
    shade = np.clip(a.ambient + lam1 * 0.62 + lam2 * 0.22, 0, 1.35)

    keep = z.min(axis=1) > 0.02
    if a.debug_tex and texids is not None:
        cols[texids >= 0] = (0.0, 1.0, 0.2)
        images = {}
    if a.browser and isink is not None:
        area = ((proj[:, 1, 0] - proj[:, 0, 0]) * (proj[:, 2, 1] - proj[:, 0, 1])
                - (proj[:, 2, 0] - proj[:, 0, 0]) * (proj[:, 1, 1] - proj[:, 0, 1]))
        keep = keep & np.where(isink == 1, area > 0, area < 0)
    elif a.cull != "none":
        area = ((proj[:, 1, 0] - proj[:, 0, 0]) * (proj[:, 2, 1] - proj[:, 0, 1])
                - (proj[:, 2, 0] - proj[:, 0, 0]) * (proj[:, 1, 1] - proj[:, 0, 1]))
        if a.cull == "front":
            keep = keep & (area < 0)
        else:
            keep = keep & (area > 0)
    depth = z.mean(axis=1)
    if a.z:
        img = zrender(proj, z, cols, shade, W, H, (26, 26, 32), keep,
                      uvs=uvs, texids=texids, images=images)
        out_img = Image.fromarray(img)
        if a.ss > 1:
            out_img = out_img.resize((a.size, a.size), Image.LANCZOS)
        out_img.save(a.out)
        print(f"[raster] {len(tris)} tris -> {a.out} ({a.view}, zbuffer)")
        return
    order = np.argsort(-depth)
    img = Image.new("RGB", (W, H), (26, 26, 32))
    dr = ImageDraw.Draw(img)
    for i in order:
        if z[i].min() <= 0.02:
            continue
        p = proj[i]
        c = cols[i] * shade[i]
        rgb = tuple(int(max(0, min(255, x * 255))) for x in c)
        dr.polygon([(p[0, 0], p[0, 1]), (p[1, 0], p[1, 1]),
                    (p[2, 0], p[2, 1])], fill=rgb)
    if a.ss > 1:
        img = img.resize((a.size, a.size), Image.LANCZOS)
    img.save(a.out)
    print(f"[raster] {len(tris)} tris -> {a.out} ({a.view})")


if __name__ == "__main__":
    main()
