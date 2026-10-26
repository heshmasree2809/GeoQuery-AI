"""Change detection: ORB/RANSAC registration, Change Vector Analysis (Lab) + Otsu, class transitions."""
import cv2
import numpy as np
from .landcover import CLASSES

SEVERITY = [(0.5, "NEGLIGIBLE"), (2.0, "LOW"), (10.0, "MEDIUM"), (25.0, "HIGH"), (101.0, "CRITICAL")]


def severity(pct: float) -> str:
    return next(label for limit, label in SEVERITY if pct < limit)


def register(a: np.ndarray, b: np.ndarray):
    """Align image b onto image a. Returns (warped_b, valid_mask, info)."""
    h, w = a.shape[:2]
    info = {"method": "ORB features + RANSAC (partial affine)", "registered": False}
    ga, gb = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(3000)
    ka, da = orb.detectAndCompute(ga, None)
    kb, db = orb.detectAndCompute(gb, None)
    identity = (b, np.ones((h, w), np.uint8))
    if da is None or db is None or len(ka) < 12 or len(kb) < 12:
        info.update(reason="Too few features for registration; assumed pre-aligned.", shift_px=0.0, inliers=0)
        return (*identity, info)
    matches = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(db, da), key=lambda m: m.distance)[:400]
    if len(matches) < 12:
        info.update(reason="Too few matches; assumed pre-aligned.", shift_px=0.0, inliers=0)
        return (*identity, info)
    src = np.float32([kb[m.queryIdx].pt for m in matches])
    dst = np.float32([ka[m.trainIdx].pt for m in matches])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000)
    n_in = int(inl.sum()) if inl is not None else 0
    if M is None or n_in < 12 or n_in / len(matches) < 0.3:
        info.update(reason="Registration unreliable; assumed pre-aligned.", shift_px=0.0, inliers=n_in)
        return (*identity, info)
    scale = float(np.hypot(M[0, 0], M[1, 0]))
    rot = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    if not (0.9 < scale < 1.1) or abs(rot) > 10:
        info.update(reason="Implausible transform rejected; assumed pre-aligned.", shift_px=0.0, inliers=n_in)
        return (*identity, info)
    c = np.array([w / 2, h / 2, 1.0])
    shift = float(np.linalg.norm(M @ c - c[:2]))
    warped = cv2.warpAffine(b, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones(b.shape[:2], np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8))
    info.update(registered=True, shift_px=round(shift, 2), rotation_deg=round(rot, 3), scale=round(scale, 4),
                inliers=n_in, matches=len(matches))
    return warped, valid, info


def cva(a, b, valid, min_lab=10.0):
    """Change Vector Analysis in Lab space with Otsu thresholding. Returns (mask, magnitude, threshold)."""
    la = cv2.cvtColor(cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    lb = cv2.cvtColor(cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32) / 255, cv2.COLOR_BGR2Lab)
    diff = lb - la
    vm = valid.astype(bool)
    for ch in range(3):  # remove global radiometric offset (illumination / season)
        diff[..., ch] -= np.median(diff[..., ch][vm])
    mag = np.sqrt((diff ** 2).sum(-1))
    mag[~vm] = 0
    u8 = np.clip(mag * 255.0 / 60.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(u8[vm].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(float(t) * 60.0 / 255.0, min_lab)  # absolute floor: no change is a valid answer
    mask = ((mag > thr) & vm).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 25:
            mask[lab == i] = 0
    return mask, mag, thr


def transitions(la, lb, mask, gsd_m):
    out = []
    for i, f in enumerate(CLASSES):
        for j, t in enumerate(CLASSES):
            if i == j:
                continue
            n = int(((la == i) & (lb == j) & (mask == 1)).sum())
            if n:
                out.append({"from": f, "to": t, "pixels": n, "area_km2": round(n * gsd_m ** 2 / 1e6, 4)})
    return sorted(out, key=lambda d: -d["pixels"])


def hotspots(mask, gsd_m, top=5, min_fraction=0.2):
    h, w = mask.shape
    cell = max(16, min(h, w) // 8)
    spots = []
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            blk = mask[y:y + cell, x:x + cell]
            frac = float(blk.mean())
            if frac >= min_fraction:
                spots.append({"bbox_px": [x, y, min(x + cell, w), min(y + cell, h)],
                              "changed_percent": round(100 * frac, 1),
                              "changed_km2": round(float(blk.sum()) * gsd_m ** 2 / 1e6, 4)})
    return sorted(spots, key=lambda d: -d["changed_percent"])[:top]


def render(img, mask, spots):
    vis = img.copy()
    red = np.zeros_like(vis)
    red[..., 2] = 255
    vis[mask == 1] = cv2.addWeighted(vis, 0.45, red, 0.55, 0)[mask == 1]
    for s in spots:
        x0, y0, x1, y1 = s["bbox_px"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 255), 1)
    return vis
