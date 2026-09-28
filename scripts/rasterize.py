"""Rasterise extracted triangles (numpy z-buffer, PIL output).

Supports a --browser mode that reproduces what three.js actually does:
mesh materials render FrontSide, Ink materials BackSide. Without that,
"looks fine to me" is meaningless.

    python3 scripts/rasterize.py --npz /tmp/f.npz --out /tmp/p.png \
        --view 3q --size 640 --browser
"""
import argparse
import math

import numpy as np
from PIL import Image

VIEWS = {
    "front": ((0.0, 3.1, 1.15), (0, 0, 1.00), 40),
    "3q": ((1.85, 2.65, 1.40), (0, 0, 1.00), 40),
    "side": ((3.1, 0.0, 1.15), (0, 0, 1.00), 40),
    "back": ((0.0, -3.1, 1.20), (0, 0, 1.05), 40),
    "head": ((0.28, 1.10, 1.72), (0, 0.02, 1.70), 36),
    "face": ((0.10, 0.58, 1.745), (0, 0.03, 1.735), 34),
    "legs": ((0.0, 2.3, 0.62), (0, 0, 0.58), 40),
}


def basis(pos, target, fov, W, H):
    pos = np.array(pos, float)
    fwd = np.array(target, float) - pos
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    return pos, right, up, fwd, (H / 2.0) / math.tan(math.radians(fov) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="3q", choices=sorted(VIEWS))
    ap.add_argument("--size", type=int, default=620)
    ap.add_argument("--ambient", type=float, default=0.34)
    ap.add_argument("--browser", action="store_true",
                    help="three.js sides: mesh FrontSide, Ink BackSide")
    ap.add_argument("--cull", default="none", choices=["none", "front", "back"])
    a = ap.parse_args()

    d = np.load(a.npz)
    tris = d["tris"].astype(np.float64)
    cols = d["cols"].astype(np.float64)
    uvs = d["uvs"].astype(np.float64) if "uvs" in d.keys() else None
    texids = d["texids"] if "texids" in d.keys() else None
    isink = d["isink"] if "isink" in d.keys() else None
    images = {int(k.split("_")[1]): d[k].astype(np.float64)
              for k in d.keys() if k.startswith("img_")}

    W = H = a.size
    pos, right, up, fwd, f = basis(*VIEWS[a.view], W, H)
    rel = tris.reshape(-1, 3) - pos
    x, y, z = rel @ right, rel @ up, rel @ fwd
    sx = W / 2.0 + f * x / np.maximum(z, 1e-6)
    sy = H / 2.0 - f * y / np.maximum(z, 1e-6)
    proj = np.stack([sx, sy], 1).reshape(-1, 3, 2)
    z = z.reshape(-1, 3)

    fn = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    tocam = pos - tris[:, 0]
    tocam /= np.maximum(np.linalg.norm(tocam, axis=1, keepdims=True), 1e-9)
    fn[(fn * tocam).sum(1) < 0] *= -1
    L1 = np.array([0.40, 0.66, 0.64]); L1 /= np.linalg.norm(L1)
    L2 = np.array([-0.60, 0.34, 0.28]); L2 /= np.linalg.norm(L2)
    shade = np.clip(a.ambient + np.clip(fn @ L1, 0, 1) * 0.64
                    + np.clip(fn @ L2, 0, 1) * 0.22, 0, 1.4)

    area = ((proj[:, 1, 0] - proj[:, 0, 0]) * (proj[:, 2, 1] - proj[:, 0, 1])
            - (proj[:, 2, 0] - proj[:, 0, 0]) * (proj[:, 1, 1] - proj[:, 0, 1]))
    keep = z.min(axis=1) > 0.02
    if a.browser and isink is not None:
        keep &= np.where(isink == 1, area > 0, area < 0)
    elif a.cull == "front":
        keep &= area < 0
    elif a.cull == "back":
        keep &= area > 0

    zbuf = np.full((H, W), 1e9)
    cbuf = np.full((H, W, 3), (26, 26, 32), dtype=np.float64)
    for i in np.argsort(-z.min(axis=1)):
        if not keep[i]:
            continue
        p, zz = proj[i], z[i]
        x0, x1 = max(int(p[:, 0].min()), 0), min(int(np.ceil(p[:, 0].max())), W - 1)
        y0, y1 = max(int(p[:, 1].min()), 0), min(int(np.ceil(p[:, 1].max())), H - 1)
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        dd = ((p[1, 0] - p[0, 0]) * (p[2, 1] - p[0, 1])
              - (p[2, 0] - p[0, 0]) * (p[1, 1] - p[0, 1]))
        if abs(dd) < 1e-9:
            continue
        w0 = ((p[1, 0] - gx) * (p[2, 1] - gy) - (p[2, 0] - gx) * (p[1, 1] - gy)) / dd
        w1 = ((p[2, 0] - gx) * (p[0, 1] - gy) - (p[0, 0] - gx) * (p[2, 1] - gy)) / dd
        w2 = 1.0 - w0 - w1
        m = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        dep = w0 * zz[0] + w1 * zz[1] + w2 * zz[2]
        subz = zbuf[y0:y1 + 1, x0:x1 + 1]
        hit = m & (dep < subz)
        if not hit.any():
            continue
        subz[hit] = dep[hit]
        tid = int(texids[i]) if texids is not None else -1
        if tid >= 0 and tid in images:
            im = images[tid]
            ih, iw = im.shape[:2]
            uvw = uvs[i]
            U = w0 * uvw[0, 0] + w1 * uvw[1, 0] + w2 * uvw[2, 0]
            V = w0 * uvw[0, 1] + w1 * uvw[1, 1] + w2 * uvw[2, 1]
            ix = np.clip((U * iw).astype(int), 0, iw - 1)
            iy = np.clip(((1 - V) * ih).astype(int), 0, ih - 1)
            c = im[iy, ix] * shade[i] * 255
        else:
            c = np.broadcast_to(np.array(cols[i]).reshape(1, 1, 3)
                                * shade[i] * 255, hit.shape + (3,))
        cbuf[y0:y1 + 1, x0:x1 + 1][hit] = np.clip(c[hit], 0, 255)

    Image.fromarray(cbuf.astype(np.uint8)).save(a.out)
    print(f"[raster] {len(tris)} tris -> {a.out} ({a.view}"
          f"{', browser' if a.browser else ''})")


if __name__ == "__main__":
    main()
