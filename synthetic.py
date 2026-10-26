"""Synthetic scenes with known ground truth, used by tests and demos."""
import cv2
import numpy as np


def _noise(img, rng, sigma=4):
    out = img.astype(np.float32) + rng.normal(0, sigma, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def harbour(size=512, n_vessels=12, seed=1):
    """Returns (BGR image, ground-truth dict). Water bottom ~60%, land strip on top."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = (110, 70, 25)                                 # water (BGR)
    land_h = int(size * 0.40)
    img[:land_h, :size // 3] = (40, 110, 50)               # vegetation
    img[:land_h, size // 3:2 * size // 3] = (90, 120, 150) # bare soil
    img[:land_h, 2 * size // 3:] = (95, 95, 98)            # built-up base (roads/shadows)
    for y in range(0, land_h, 16):                         # built-up mosaic
        for x in range(2 * size // 3, size, 16):
            v = int(rng.integers(120, 215))
            img[y:y + 14, x:x + 14] = (v, v, v + 4)
    boxes = []
    tries = 0
    while len(boxes) < n_vessels and tries < 1000:
        tries += 1
        cx, cy = int(rng.integers(30, size - 30)), int(rng.integers(land_h + 40, size - 30))
        if any(abs(cx - x) < 40 and abs(cy - y) < 40 for x, y in boxes):
            continue
        boxes.append((cx, cy))
        ang = float(rng.uniform(0, 180))
        pts = cv2.boxPoints(((cx, cy), (16, 5), ang)).astype(np.int32)
        cv2.fillPoly(img, [pts], (225, 228, 232))
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return _noise(img, rng), {"vessels": len(boxes), "land_rows": land_h}


def change_pair(size=512, shift=(4, 3), seed=2):
    """Date A: mixed fields. Date B: shifted, brighter, with a vegetation patch turned to bare soil."""
    rng = np.random.default_rng(seed)
    a = np.zeros((size, size, 3), np.uint8)
    a[:] = (45, 115, 55)
    for _ in range(140):                                   # texture: fields/trees for ORB features
        x, y = rng.integers(0, size - 24, 2)
        w, h = rng.integers(6, 24, 2)
        g = int(rng.integers(85, 135))
        a[y:y + h, x:x + w] = (int(rng.integers(30, 60)), g, int(rng.integers(40, 70)))
    a[:, size * 3 // 4:] = (110, 125, 140)                 # static bare/urban strip
    a = _noise(a, rng, 3)
    b = a.copy()
    p0, p1 = size // 5, size // 5 + size // 5
    b[p0:p1, p0:p1] = (95, 125, 150)                       # veg -> bare soil patch
    b = np.clip(b.astype(np.int16) + 6, 0, 255).astype(np.uint8)
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    b = cv2.warpAffine(b, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
    truth = {"patch_fraction": ((p1 - p0) ** 2) / size ** 2, "shift_px": float(np.hypot(*shift))}
    return a, b, truth
