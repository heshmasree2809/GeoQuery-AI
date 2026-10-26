"""Executes a plan with deterministic analysis code and assembles answer + evidence trail."""
import cv2
import numpy as np
from . import change, detection, indices, landcover as lc, planner
from .imaging import to_data_uri


class _Ctx:
    def __init__(self, a, b, nir, gsd):
        self.a, self.b, self.nir, self.gsd = a, b, nir, gsd
        self._model = self._labels = None
        self.px = a.shape[0] * a.shape[1]
        self.evidence, self.trail, self.say, self.findings = [], [], [], {}

    def landcover(self):
        if self._labels is None:
            self._model = lc.LandCoverModel().fit(self.a)
            self._labels = self._model.predict(self.a)
        return self._model, self._labels

    def ev(self, title, img, method):
        self.evidence.append({"title": title, "method": method, "image": to_data_uri(img)})

    def km2(self, px):
        return px * self.gsd ** 2 / 1e6


def _landcover(c, intents):
    _, labels = c.landcover()
    s = lc.summarize(labels, c.gsd)
    c.findings["landcover"] = s
    c.ev("Land-cover map", lc.overlay(c.a, labels), "MiniBatch k-means (k=6) + spectral labelling rules")
    parts = [f"{k.replace('_', ' ')} {v['percent']}%" for k, v in sorted(s.items(), key=lambda kv: -kv[1]["pixels"]) if v["pixels"]]
    total = c.km2(c.px)
    if "vessels" not in intents and "vegetation" not in intents or "landcover" in intents:
        w = s["water"]
        c.say.append(f"{w['percent']}% of the scene is water ({w['area_km2']} km² of {total:.2f} km²). Composition: {', '.join(parts)}.")
    for k, v in s.items():
        if v["pixels"] and k in ("water", "vegetation"):
            c.trail.append({"claim": f"{k.replace('_', ' ')} covers {v['percent']}% ({v['area_km2']} km²)",
                            "pixels": f"{v['pixels']} of {c.px}", "measurement": f"pixels × GSD² = {v['pixels']} × {c.gsd:.3g}² m²",
                            "method": "k-means clusters labelled by spectral rules", "evidence": "Land-cover map"})


def _indices(c, plan):
    summary, arrays = indices.compute(c.a, c.nir)
    c.findings["indices"] = summary
    if summary["ndvi"]["available"]:
        n = summary["ndvi"]
        c.say.append(f"NDVI mean {n['mean']}; {n['vegetated_percent']}% of pixels exceed 0.3 (vegetated).")
        c.ev("NDVI", indices.heatmap(arrays["ndvi"], -0.2, 0.8), "NDVI = (NIR − Red) / (NIR + Red)")
        c.trail.append({"claim": f"{n['vegetated_percent']}% vegetated by NDVI>0.3", "pixels": f"{n['vegetated_pixels']} of {c.px}",
                        "measurement": "per-pixel NDVI", "method": "NIR/Red band ratio", "evidence": "NDVI"})
    else:
        if plan.explicit_index:
            c.say.append("NDVI/NDWI are UNAVAILABLE: they need a near-infrared band and this input is RGB-only. "
                         "I will not substitute a guess; a visible-band greenness proxy (Excess Green) is shown instead.")
        c.ev("Excess Green (RGB proxy, not NDVI)", indices.heatmap(arrays["exg"], -0.2, 0.5), "ExG = 2G − R − B")
        c.trail.append({"claim": "NDVI unavailable", "pixels": "n/a", "measurement": "none",
                        "method": summary["ndvi"]["reason"], "evidence": "Excess Green (RGB proxy, not NDVI)"})
    veg = c.findings.get("landcover", {}).get("vegetation")
    if veg and not plan.explicit_index:
        c.say.append(f"Vegetation covers {veg['percent']}% of the scene ({veg['area_km2']} km²) by RGB land-cover classification.")


def _vessels(c):
    _, labels = c.landcover()
    ves = detection.detect_vessels(labels == 0, c.gsd)
    c.findings["vessels"] = {"count": len(ves), "items": [{k: v for k, v in d.items() if k != "_box"} for d in ves[:100]]}
    if ves:
        L = [d["length_m"] for d in ves]
        c.say.append(f"{len(ves)} vessels detected; estimated lengths {min(L):.0f}–{max(L):.0f} m (median {float(np.median(L)):.0f} m).")
    else:
        c.say.append("No vessels detected on open water at this resolution.")
    c.ev("Detected vessels", detection.draw(c.a, ves), "Objects forming holes in the water mask, verified by a water ring")
    c.trail.append({"claim": f"{len(ves)} vessels", "pixels": f"{sum(d['pixels'] for d in ves)} object pixels",
                    "measurement": "length = long side of min-area rectangle × GSD", "method": "morphological closing of water mask + ring check",
                    "evidence": "Detected vessels"})


def _change(c):
    warped, valid, reg = change.register(c.a, c.b)
    mask, mag, thr = change.cva(c.a, warped, valid)
    n_valid = max(int(valid.sum()), 1)
    px = int(mask.sum())
    pct = 100.0 * px / n_valid
    sev = change.severity(pct)
    spots = change.hotspots(mask, c.gsd)
    c.findings["registration"] = reg
    c.findings["change"] = {"changed_pixels": px, "changed_percent": round(pct, 2), "changed_km2": round(c.km2(px), 4),
                            "severity": sev, "threshold_lab": round(thr, 2), "hotspots": spots}
    if reg["registered"]:
        c.say.append(f"Registration recovered a {reg['shift_px']} px shift ({reg['inliers']} RANSAC inliers).")
    else:
        c.say.append(f"Registration note: {reg['reason']}")
    c.say.append(f"{pct:.2f}% of the scene changed ({c.km2(px):.3f} km²), severity {sev}.")
    c.ev("Change map", change.render(c.a, mask, spots), "Change Vector Analysis (Lab) + Otsu; yellow boxes = hotspots")
    c.ev("Change magnitude", indices.heatmap(mag, 0, 40), "Per-pixel Lab colour distance after alignment")
    c.trail.append({"claim": f"{pct:.2f}% changed, {sev}", "pixels": f"{px} of {n_valid} valid",
                    "measurement": f"Lab distance > {thr:.1f}", "method": "CVA + Otsu (with absolute floor)", "evidence": "Change map"})
    c._warped, c._mask = warped, mask


def _transitions(c):
    model, la = c.landcover()
    lb = model.predict(c._warped)
    tr = change.transitions(la, lb, c._mask, c.gsd)
    c.findings["transitions"] = tr[:10]
    if tr:
        t = tr[0]
        c.say.append(f"Largest transition: {t['from'].replace('_', ' ')} → {t['to'].replace('_', ' ')}, {t['area_km2']} km².")
        c.trail.append({"claim": f"{t['from']} → {t['to']} {t['area_km2']} km²", "pixels": f"{t['pixels']}",
                        "measurement": "changed pixels whose class differs between dates", "method": "same k-means model applied to both dates",
                        "evidence": "Change map"})
    spots = c.findings["change"]["hotspots"]
    if spots:
        c.say.append(f"{len(spots)} hotspot(s); strongest at pixel box {spots[0]['bbox_px']} ({spots[0]['changed_percent']}% changed).")
    c.ev("Date B classified", lc.overlay(c._warped, lb), "Same land-cover model applied to aligned date B")


def run(question, img_a, img_b=None, nir=None, gsd_m=10.0):
    plan = planner.make_plan(question, img_b is not None, nir is not None)
    if img_b is not None and img_b.shape[:2] != img_a.shape[:2]:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)
        plan.warnings.append("Image B was resized to match image A; check the pixel size is comparable.")
    c = _Ctx(img_a, img_b, nir, gsd_m)
    for step in plan.steps:
        t = step["tool"]
        if t == "landcover":
            _landcover(c, plan.intents)
        elif t == "indices":
            _indices(c, plan)
        elif t == "vessel_detection":
            _vessels(c)
        elif t == "change_detection":
            _change(c)
        elif t == "transition_analysis":
            _transitions(c)
    answer = " ".join(c.say) if c.say else "No analysis could be run for this question."
    if plan.warnings:
        answer += " ⚠ " + " ".join(plan.warnings)
    return {"question": question, "answer": answer, "plan": plan.steps, "findings": c.findings,
            "evidence": c.evidence, "evidence_trail": c.trail, "warnings": plan.warnings,
            "scene": {"pixels": c.px, "gsd_m": gsd_m, "area_km2": round(c.km2(c.px), 3)}}
