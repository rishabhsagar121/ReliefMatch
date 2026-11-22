from resource_matcher import (
recommend_resources,
match_all_pending_requests
)

inventory = [
{"Resource": "Drinking Water", "Location": "Meerut", "Available": 60},
{"Resource": "Drinking Water", "Location": "Ghaziabad", "Available": 100},
{"Resource": "Food Packets", "Location": "Noida", "Available": 200}
]

request = {
"ID": "REQ-004",
"Emergency": "Flood",
"Location": "Meerut",
"Resource": "Drinking Water",
"Quantity": 100,
"Priority": "High",
"Status": "Pending"
}

result = recommend_resources(request, inventory)

assert result["status"] == "Fully Matchable"
assert result["requested_quantity"] == 100
assert result["allocated_quantity"] == 100
assert result["remaining_quantity"] == 0

assert result["recommendations"][0]["Location"] == "Meerut"
assert result["recommendations"][0]["Recommended Quantity"] == 60
assert result["recommendations"][1]["Location"] == "Ghaziabad"
assert result["recommendations"][1]["Recommended Quantity"] == 40

requests = [
{
"ID": "REQ-005",
"Resource": "Drinking Water",
"Location": "Meerut",
"Quantity": 80,
"Priority": "Low",
"Status": "Pending"
},
{
"ID": "REQ-006",
"Resource": "Drinking Water",
"Location": "Meerut",
"Quantity": 70,
"Priority": "Critical",
"Status": "Pending"
}
]

shared_inventory = [
{"Resource": "Drinking Water", "Location": "Meerut", "Available": 100}
]

original_inventory = [dict(item) for item in shared_inventory]

results = match_all_pending_requests(requests, shared_inventory)

critical_result = next(
result for result in results
if result["Request ID"] == "REQ-006"
)

low_result = next(
result for result in results
if result["Request ID"] == "REQ-005"
)

assert critical_result["Recommended Quantity"] == 70
assert critical_result["Matching Status"] == "Fully Matchable"

assert low_result["Recommended Quantity"] == 30
assert low_result["Remaining Quantity"] == 50
assert low_result["Matching Status"] == "Partially Matchable"

assert shared_inventory == original_inventory

print("All resource matcher tests passed!")
print("Single-request matching: PASS")
print("Location preference: PASS")
print("Priority-based stock sharing: PASS")
print("Partial allocation: PASS")
print("Original inventory protection: PASS")
