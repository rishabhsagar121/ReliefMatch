import smart_priority
from smart_priority import assess_request, ml_status, reset_ml_cache


class FakeML:
    """Stands in for ml/ml_engine.py so the test needs no model file."""

    def __init__(self, category, confidence, review=False):
        self.out = {"category": category, "confidence": confidence, "requires_review": review}

    def classify_request(self, message):
        return self.out


def use(category, confidence, review=False):
    reset_ml_cache()
    smart_priority._ml_module = FakeML(category, confidence, review)


# Confident ML result raises a low rule priority
use("injured_or_dead_people", 0.9)
r = assess_request("Other", "Some people hurt near the bridge", "Medicines", 5)
assert r["priority"] == "Critical" and r["ml_applied"] and r["ml"]["label"] == "Injured or dead people"

# ML never lowers a Critical rule-based priority
use("sympathy_and_support", 0.95)
r = assess_request("Flood", "Family trapped on the roof", "Rescue Boats", 2)
assert r["priority"] == "Critical" and not r["ml_applied"]

# Low-confidence ML result is shown but does not change the priority
use("injured_or_dead_people", 0.40, review=True)
r = assess_request("Food Shortage", "Village needs extra food packets", "Food Packets", 40)
assert r["priority"] == r["rule_priority"] and r["ml"]["requires_review"] and not r["ml_applied"]

# Empty description: rules only, with a note
r = assess_request("Flood", "", "Drinking Water", 10)
assert r["ml"] is None and r["ml_note"]

# No model available: app still works with rules only
reset_ml_cache()
smart_priority._ml_reason = "AI model file not found."
r = assess_request("Flood", "Family trapped on roof", "Rescue Boats", 2)
assert r["priority"] == "Critical" and r["ml"] is None and "not found" in r["ml_note"]
assert ml_status() == (False, "AI model file not found.")


# classify_batch: one result per message, blank messages handled
from smart_priority import classify_batch
use("requests_or_urgent_needs", 0.8)
batch = classify_batch(["need water", "", "need food"])
assert len(batch) == 3 and batch[0]["label"] == "Requests or urgent needs"
assert batch[0]["ml_priority"] == "High" and batch[1]["requires_review"] and batch[1]["category"] == ""

# classify_batch without a model raises a clear error
reset_ml_cache()
smart_priority._ml_reason = "AI model file not found."
try:
    classify_batch(["x"])
    raise AssertionError("expected RuntimeError")
except RuntimeError as error:
    assert "not found" in str(error)
reset_ml_cache()
print("All smart priority tests passed!")