"""Offline natural-language planner: maps a question to an ordered list of deterministic analysis tools."""
import re
from dataclasses import dataclass, field

P = lambda s: re.compile(s, re.I)
CHANGE = P(r"chang|differen|compar|before and after|between (the |these )?(two|dates)|transition|deforest|expan|new construction|what.s new")
VESSEL = P(r"vessel|ship|boat|tanker|ferry|container ship|maritime|fleet")
VEG = P(r"vegetat|ndvi|green|crop|forest|plant|canopy|farmland")
INDEX = P(r"ndvi|ndwi|index|indices|spectral")
WATER = P(r"water|sea\b|lake|river|ocean|flood|coast|harbou?r|bay\b")
COVER = P(r"land.?cover|land.?use|classif|composition|breakdown|built|urban|soil|bare|what.s in|what is in|describe|overview|summar")
UNSUPPORTED = P(r"(how many|count|detect|find|locate).{0,30}(car|vehicle|truck|aircraft|plane|building|house|tree|person|people|road)")


@dataclass
class Plan:
    steps: list = field(default_factory=list)
    intents: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    explicit_index: bool = False

    def add(self, tool, why):
        if all(s["tool"] != tool for s in self.steps):
            self.steps.append({"tool": tool, "why": why})


def make_plan(question: str, has_b: bool, has_nir: bool) -> Plan:
    q = question or ""
    p = Plan(explicit_index=bool(INDEX.search(q)))
    if UNSUPPORTED.search(q):
        p.warnings.append("Counting this object class is not supported (supported: vessels). Nothing was estimated.")
    if CHANGE.search(q):
        p.intents.append("change")
        if has_b:
            p.add("registration", "Align the two dates so differences are real, not camera shift.")
            p.add("change_detection", "Change Vector Analysis + Otsu threshold on the aligned pair.")
            p.add("transition_analysis", "Classify both dates and tabulate what turned into what.")
            return p
        p.warnings.append("Change questions need a second image (date B); none was provided.")
    if VESSEL.search(q):
        p.intents.append("vessels")
        p.add("landcover", "Water mask is needed to look for objects on open water.")
        p.add("vessel_detection", "Find and measure objects inside the water mask.")
    if VEG.search(q) or INDEX.search(q):
        p.intents.append("vegetation")
        p.add("landcover", "Classify the scene for vegetation extent.")
        p.add("indices", "Compute spectral indices (UNAVAILABLE bands are reported, not faked).")
    if WATER.search(q) or COVER.search(q) or not p.steps:
        p.intents.append("landcover")
        p.add("landcover", "Cluster pixels and label clusters with spectral rules.")
    return p
