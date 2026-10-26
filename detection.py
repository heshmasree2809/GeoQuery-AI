"""Vessel detection: objects that form 'holes' inside the water mask, verified by their surroundings."""
import cv2
import numpy as np

MAX_VESSELS = 500


def detect_vessels(water: np.ndarray, gsd_m: float, max_length_m: float = 400.0, min_area_m2: float = 20.0):
    wm = water.astype(np.uint8)
    k = int(np.clip(np.ceil(60.0 / gsd_m) + 2, 5, 41)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    closed = cv2.morphologyEx(wm, cv2.MORPH_CLOSE, kernel)
    cand = ((closed == 1) & (wm == 0)).astype(np.uint8)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, cents = cv2.connectedComponentsWithStats(cand, connectivity=8)
    h, w = wm.shape
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        area_m2 = area * gsd_m ** 2
        if area_m2 < min_area_m2 or area < 4:
            continue
        pad = 6
        x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + bw + pad, w), min(y + bh + pad, h)
        comp = (lab[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, np.ones((7, 7), np.uint8)) - comp
        ring_water = float(wm[y0:y1, x0:x1][ring == 1].mean()) if ring.any() else 0.0
        if ring_water < 0.8:
            continue  # touches land -> pier/quay/shoreline, not an open-water vessel
        pts = cv2.findNonZero(comp)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        length, width = max(rw, rh) * gsd_m, min(rw, rh) * gsd_m
        if length > max_length_m:
            continue
        box = cv2.boxPoints(cv2.minAreaRect(pts)) + np.array([x0, y0])
        out.append({"id": len(out) + 1, "centroid_px": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
                    "length_m": round(float(length), 1), "width_m": round(float(width), 1),
                    "area_m2": round(float(area_m2), 1), "pixels": int(area),
                    "confidence": round(ring_water, 2), "_box": box})
    out.sort(key=lambda d: -d["area_m2"])
    return out[:MAX_VESSELS]


def draw(img, vessels):
    vis = img.copy()
    for v in vessels:
        cv2.polylines(vis, [v["_box"].astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
    return vis
