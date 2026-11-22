from priority_classifier import classify_priority, classify_priority_with_reasons

# Immediate danger -> Critical, whatever the type
assert classify_priority("Flood", "Family trapped on the roof, water rising fast", "Rescue Boats", 3) == "Critical"
assert classify_priority("Medical Emergency", "Patient unconscious and bleeding", "First Aid Kits", 2) == "Critical"

# Vulnerable people raise priority
assert classify_priority("Food Shortage", "Children and elderly have no food for 48 hours", "Food Packets", 80) == "High"

# Plain, low-risk request stays low/medium
assert classify_priority("Other", "Need some supplies for a relief camp", "Medicines", 10) in ("Low", "Medium")
assert classify_priority("Food Shortage", "Village needs extra food packets", "Food Packets", 40) == "Medium"

# Rescue resources are at least High
assert classify_priority("Other", "Need equipment", "Rescue Equipment", 5) in ("High", "Critical")

# Large quantities bump the level up
assert classify_priority("Flood", "Water supply cut", "Drinking Water", 800) == "High"

# Bad input never crashes
assert classify_priority(None, None, None, "abc") in ("Low", "Medium", "High", "Critical")

label, reasons = classify_priority_with_reasons("Flood", "Children stranded", "Rescue Boats", 20)
assert label == "Critical" and reasons

print("All priority classifier tests passed!")