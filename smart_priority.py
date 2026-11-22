"""Combines the rule-based priority with the teammate's ML classifier.

How the two are combined
------------------------
1. priority_classifier.py gives a rule-based priority (Low/Medium/High/Critical).
2. ml/ml_engine.py (the ML classifier) gives a message category and a confidence.
3. The category is mapped to a priority (CATEGORY_PRIORITY below).
4. The final priority is the HIGHER of the two, but only when the ML result is
   confident (not flagged "requires_review"). The ML can therefore raise a
   priority, never lower it, and an uncertain prediction never changes it.

If the ML model or its packages are missing, everything still works with the
rules alone, and a note explains why the AI part is switched off.
"""

from priority_classifier import LEVELS, classify_priority_with_reasons

# Starting point for this mapping - please review it with your teammate.
CATEGORY_PRIORITY = {
    "injured_or_dead_people": "Critical",
    "missing_or_found_people": "High",
    "displaced_people_and_evacuations": "High",
    "requests_or_urgent_needs": "High",
    "infrastructure_and_utility_damage": "Medium",
    "rescue_volunteering_or_donation_effort": "Medium",
    "other_relevant_information": "Low",
    "caution_and_advice": "Low",
    "sympathy_and_support": "Low",
    "not_humanitarian": "Low",
}

_ml_module = None
_ml_reason = None


def _get_ml():
    """Return (ml_engine module, None) or (None, reason it is unavailable)."""
    global _ml_module, _ml_reason

    if _ml_module is not None:
        return _ml_module, None
    if _ml_reason:
        return None, _ml_reason

    try:
        from ml import ml_engine

        ml_engine.load_model()
        _ml_module = ml_engine
        return _ml_module, None
    except FileNotFoundError:
        _ml_reason = (
            "AI model file not found. Put emergency_classifier.joblib in ml/models/."
        )
    except ImportError as error:
        _ml_reason = (
            f"AI packages are missing ({error}). Run: pip install -r requirements.txt"
        )
    except Exception as error:
        _ml_reason = f"The AI model could not be loaded: {error}"

    return None, _ml_reason


def reset_ml_cache():
    """Forget a cached load result (used by tests)."""
    global _ml_module, _ml_reason
    _ml_module, _ml_reason = None, None


def ml_status():
    """Return (is_available, message) for showing the AI status in the UI."""
    module, reason = _get_ml()
    if module is None:
        return False, reason
    return True, "AI classifier connected"


def pretty_category(category):
    return str(category).replace("_", " ").capitalize()


def assess_request(emergency, description="", resource="", quantity=0):
    """Return the final priority plus the details behind it.

    Result keys: priority, rule_priority, reasons, ml, ml_applied, ml_note
    where ml is None or {category, label, confidence, requires_review, ml_priority}.
    """
    rule_priority, rule_reasons = classify_priority_with_reasons(
        emergency, description, resource, quantity
    )
    result = {
        "priority": rule_priority,
        "rule_priority": rule_priority,
        "reasons": list(rule_reasons),
        "ml": None,
        "ml_applied": False,
        "ml_note": None,
    }

    text = str(description or "").strip()
    if not text:
        result["ml_note"] = "Enter a description to get an AI category."
        return result

    module, reason = _get_ml()
    if module is None:
        result["ml_note"] = reason
        return result

    try:
        prediction = module.classify_request(text)
    except Exception as error:
        result["ml_note"] = f"AI classification failed: {error}"
        return result

    category = prediction["category"]
    ml_priority = CATEGORY_PRIORITY.get(category, "Low")
    result["ml"] = {
        "category": category,
        "label": pretty_category(category),
        "confidence": float(prediction["confidence"]),
        "requires_review": bool(prediction["requires_review"]),
        "ml_priority": ml_priority,
    }

    if (
        not prediction["requires_review"]
        and LEVELS.index(ml_priority) > LEVELS.index(rule_priority)
    ):
        result["priority"] = ml_priority
        result["ml_applied"] = True
        result["reasons"].append(
            f"AI category '{pretty_category(category)}' raised the priority"
        )

    return result


def classify_batch(messages):
    """Classify many messages with the ML model.

    Returns one dict per message: category, label, confidence, requires_review,
    ml_priority. Raises RuntimeError if the AI model is not available.
    """
    module, reason = _get_ml()
    if module is None:
        raise RuntimeError(reason)

    results = []
    for message in messages:
        text = str(message or "").strip()
        if not text:
            results.append({
                "category": "", "label": "", "confidence": 0.0,
                "requires_review": True, "ml_priority": "Low",
            })
            continue

        prediction = module.classify_request(text)
        category = prediction["category"]
        results.append({
            "category": category,
            "label": pretty_category(category),
            "confidence": float(prediction["confidence"]),
            "requires_review": bool(prediction["requires_review"]),
            "ml_priority": CATEGORY_PRIORITY.get(category, "Low"),
        })
    return results