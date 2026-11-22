"""Rule-based priority classifier for ReliefMatch emergency requests.

Returns one of: "Low", "Medium", "High", "Critical" - the same labels that
resource_matcher.py already understands, so no other file needs to change.

This is a transparent first version. A machine-learning model can replace
classify_priority() later as long as it returns the same four labels.
"""

import re

LEVELS = ["Low", "Medium", "High", "Critical"]

# Starting level for each emergency type (index into LEVELS)
BASE_LEVEL = {
    "flood": 1,
    "earthquake": 2,
    "cyclone": 1,
    "fire": 2,
    "medical emergency": 2,
    "food shortage": 1,
    "other": 1,
}

# Any of these means people may die without immediate help -> Critical
CRITICAL_PATTERNS = [
    r"\btrapped\b", r"\bstuck\b", r"\bdrown\w*", r"\bcollaps\w*",
    r"\bunconscious\b", r"\bnot breathing\b", r"\bbleeding\b", r"\bsevere\w*",
    r"\blife[- ]threatening\b", r"\bdying\b", r"\bdeaths?\b", r"\bdead\b",
    r"\bmissing\b", r"\bburn\w*", r"\bheart attack\b", r"\bstrand\w*",
    r"\brising fast\b", r"\bswept\b", r"\bburied\b", r"\bcritical\b",
]

# Signs of vulnerable people or serious shortage -> raises priority
HIGH_PATTERNS = [
    r"\bchild\w*", r"\binfants?\b", r"\bbab(y|ies)\b", r"\belderly\b",
    r"\bpregnan\w*", r"\bdisabled\b", r"\binjur\w*", r"\bhospital\b",
    r"\bevacuat\w*", r"\burgent\w*", r"\bno (drinking )?water\b",
    r"\bno food\b", r"\bwithout (food|water|shelter|medicine)\b",
    r"\b(24|48|72) hours\b", r"\bdays\b", r"\bshelter\b", r"\bhomeless\b",
    r"\bcut off\b", r"\bisolated\b", r"\bdisease\b", r"\boutbreak\b",
]

RESCUE_RESOURCES = {"rescue boats", "rescue equipment"}
LARGE_QUANTITY = 500


def _matches(patterns, text):
    return [p for p in patterns if re.search(p, text)]


def classify_priority_with_reasons(emergency, description="", resource="", quantity=0):
    """Return (priority_label, list_of_reasons)."""
    emergency_key = str(emergency or "").strip().casefold()
    resource_key = str(resource or "").strip().casefold()
    text = str(description or "").casefold()

    try:
        quantity = int(quantity)
    except (TypeError, ValueError):
        quantity = 0

    level = BASE_LEVEL.get(emergency_key, BASE_LEVEL["other"])
    reasons = [f"Emergency type: {emergency or 'Other'}"]

    critical_hits = _matches(CRITICAL_PATTERNS, text)
    high_hits = _matches(HIGH_PATTERNS, text)

    if critical_hits:
        level = 3
        reasons.append("Description suggests immediate danger to life")
    else:
        if high_hits:
            level += 1
            reasons.append("Description mentions vulnerable people or serious shortage")
        if len(high_hits) >= 3:
            level += 1
            reasons.append("Several urgent indicators in the description")
        # Only signs of immediate danger can make a request Critical on their own
        level = min(level, 2)

    if resource_key in RESCUE_RESOURCES and level < 2:
        level = 2
        reasons.append("Rescue resources requested")

    if quantity >= LARGE_QUANTITY:
        level += 1
        reasons.append(f"Large quantity requested ({quantity})")

    level = max(0, min(level, 3))
    return LEVELS[level], reasons


def classify_priority(emergency, description="", resource="", quantity=0):
    """Return just the priority label: Low, Medium, High or Critical."""
    return classify_priority_with_reasons(emergency, description, resource, quantity)[0]