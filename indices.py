"""Spectral indices. Indices needing bands that are absent are reported UNAVAILABLE, never faked."""
import cv2
import numpy as np


def compute(img: np.ndarray, nir: np.ndarray | None):
    f = img[..., ::-1].astype(np.float32) / 255.0
    r, g, b = f[..., 0], f[..., 1], f[..., 2]
    summary, arrays = {}, {}
    if nir is None:
        why = "Requires a near-infrared band; the input is RGB-only."
        summary["ndvi"] = {"available": False, "status": "UNAVAILABLE", "reason": why}
        summary["ndwi"] = {"available": False, "status": "UNAVAILABLE", "reason": why}
    else:
        n = nir.astype(np.float32) / 255.0
        ndvi = (n - r) / (n + r + 1e-6)
        ndwi = (g - n) / (g + n + 1e-6)
        arrays["ndvi"], arrays["ndwi"] = ndvi, ndwi
        summary["ndvi"] = {"available": True, "mean": round(float(ndvi.mean()), 4),
                           "vegetated_pixels": int((ndvi > 0.3).sum()),
                           "vegetated_percent": round(100 * float((ndvi > 0.3).mean()), 2)}
        summary["ndwi"] = {"available": True, "mean": round(float(ndwi.mean()), 4),
                           "water_pixels": int((ndwi > 0).sum()),
                           "water_percent": round(100 * float((ndwi > 0).mean()), 2)}
    exg = 2 * g - r - b
    arrays["exg"] = exg
    summary["exg"] = {"available": True, "proxy": True, "mean": round(float(exg.mean()), 4),
                      "note": "Excess Green is a visible-band greenness PROXY, not NDVI."}
    return summary, arrays


def heatmap(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    norm = np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
