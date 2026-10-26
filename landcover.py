"""Land-cover classification: MiniBatch k-means clustering + physics-based labelling rules."""
import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

CLASSES = ["water", "vegetation", "bare_soil", "built_up", "cloud", "other"]
COLORS_BGR = {
    "water": (200, 120, 30), "vegetation": (60, 170, 60), "bare_soil": (80, 140, 200),
    "built_up": (150, 150, 150), "cloud": (255, 255, 255), "other": (200, 60, 200),
}
TEX_SCALE = 4.0  # texture is up-weighted in clustering space


def features(img: np.ndarray) -> np.ndarray:
    f = img[..., ::-1].astype(np.float32) / 255.0  # RGB
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    mx, mn = f.max(-1), f.min(-1)
    sat = (mx - mn) / (mx + 1e-6)
    gray = f.mean(-1)
    m = cv2.blur(gray, (5, 5))
    tex = np.sqrt(np.maximum(cv2.blur(gray * gray, (5, 5)) - m * m, 0))
    exg = 2 * g - r - b
    return np.dstack([r, g, b, exg, sat, tex * TEX_SCALE]).astype(np.float32)


def classify_center(c) -> str:
    """Physics/spectral rules applied to a cluster centre (r, g, b, ExG, saturation, texture)."""
    r, g, b, exg, sat, tex = [float(x) for x in c]
    tex /= TEX_SCALE
    bright = (r + g + b) / 3
    if bright > 0.80 and sat < 0.15 and tex < 0.05:
        return "cloud"
    if g > b + 0.03 and g > r + 0.01 and exg > 0.04:
        return "vegetation"          # chlorophyll: green reflectance peak
    if b > r + 0.05 and b >= g - 0.05 and bright < 0.60 and tex < 0.10:
        return "water"               # clearly blue-dominant, low albedo, smooth
    if bright < 0.18 and tex < 0.04:
        return "water"               # very dark, spatially flat
    if r >= g and g >= b - 0.01 and sat > 0.12:
        return "bare_soil"           # red >= green >= blue, moderately saturated
    if sat < 0.25:
        return "built_up"            # grey, impervious surfaces
    return "other"


class LandCoverModel:
    def __init__(self, k: int = 6, seed: int = 0):
        self.km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
        self.cluster_class = None

    def fit(self, img):
        x = features(img).reshape(-1, 6)
        rng = np.random.default_rng(0)
        sample = x[rng.choice(len(x), size=min(len(x), 60000), replace=False)]
        self.km.fit(sample)
        self.cluster_class = np.array([CLASSES.index(classify_center(c)) for c in self.km.cluster_centers_], np.uint8)
        return self

    def predict(self, img) -> np.ndarray:
        h, w = img.shape[:2]
        idx = self.km.predict(features(img).reshape(-1, 6))
        return self.cluster_class[idx].reshape(h, w)


def summarize(labels: np.ndarray, gsd_m: float) -> dict:
    total = labels.size
    out = {}
    for i, name in enumerate(CLASSES):
        n = int((labels == i).sum())
        out[name] = {"pixels": n, "percent": round(100.0 * n / total, 2),
                     "area_km2": round(n * gsd_m ** 2 / 1e6, 4)}
    return out


def overlay(img, labels, alpha=0.5):
    color = np.zeros_like(img)
    for i, name in enumerate(CLASSES):
        color[labels == i] = COLORS_BGR[name]
    return cv2.addWeighted(img, 1 - alpha, color, alpha, 0)
