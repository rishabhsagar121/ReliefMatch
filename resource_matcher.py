"""Resource matching logic for ReliefMatch.

Public functions (names, arguments and return formats are unchanged):
    - recommend_resources(request, inventory)
    - match_all_pending_requests(requests, inventory)
"""

PRIORITY_ORDER = {
    "critical": 0,
    "high": 1,
    "medium": 2,
    "low": 3,
}
DEFAULT_PRIORITY_RANK = 4  # "Pending Classification", blanks, unknown values


def _safe_int(value, default=0):
    """Convert value to int without raising (handles None, '', 'abc', 10.0)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def _normalize(value):
    return str(value if value is not None else "").strip().casefold()


def recommend_resources(request, inventory):
    """Recommend inventory rows for a single request.

    Rows in the same location as the request are used first, then rows with
    the most stock. The inventory passed in is never modified.
    """
    request_id = request.get("ID", "Unknown")
    required_resource = _normalize(request.get("Resource", ""))
    required_quantity = _safe_int(request.get("Quantity", 0))
    request_location = _normalize(request.get("Location", ""))

    if not required_resource or required_quantity <= 0:
        return {
            "request_id": request_id,
            "status": "Invalid Request",
            "requested_quantity": required_quantity,
            "allocated_quantity": 0,
            "remaining_quantity": max(required_quantity, 0),
            "recommendations": [],
        }

    available_inventory = []

    for item in inventory:
        if _normalize(item.get("Resource", "")) != required_resource:
            continue

        available_quantity = _safe_int(item.get("Available", 0))
        if available_quantity <= 0:
            continue

        location = str(item.get("Location", "")).strip()

        available_inventory.append({
            "Resource": item.get("Resource", ""),
            "Location": location,
            "Available": available_quantity,
            "location_match": location.casefold() == request_location,
        })

    # 1) same-location stock first, 2) larger stock first, 3) alphabetical
    available_inventory.sort(
        key=lambda row: (
            not row["location_match"],
            -row["Available"],
            row["Location"].casefold(),
        )
    )

    recommendations = []
    remaining = required_quantity

    for row in available_inventory:
        if remaining <= 0:
            break

        allocated = min(row["Available"], remaining)

        recommendations.append({
            "Resource": row["Resource"],
            "Location": row["Location"],
            "Available": row["Available"],
            "Recommended Quantity": allocated,
        })

        remaining -= allocated

    allocated_total = required_quantity - remaining

    if remaining == 0:
        status = "Fully Matchable"
    elif allocated_total > 0:
        status = "Partially Matchable"
    else:
        status = "Unavailable"

    return {
        "request_id": request_id,
        "status": status,
        "requested_quantity": required_quantity,
        "allocated_quantity": allocated_total,
        "remaining_quantity": remaining,
        "recommendations": recommendations,
    }


def _deduct_from_inventory(working_inventory, recommendations):
    """Subtract recommended quantities from a working copy of the inventory."""
    for rec in recommendations:
        to_deduct = rec["Recommended Quantity"]
        resource = _normalize(rec["Resource"])
        location = _normalize(rec["Location"])

        for row in working_inventory:
            if to_deduct <= 0:
                break
            if (
                _normalize(row.get("Resource", "")) == resource
                and _normalize(row.get("Location", "")) == location
            ):
                current = _safe_int(row.get("Available", 0))
                used = min(current, to_deduct)
                row["Available"] = current - used
                to_deduct -= used


def match_all_pending_requests(requests, inventory):
    """Match every pending request against the inventory.

    Stock is shared fairly: requests are processed by priority (High first,
    then by original order), and stock used by one request is not offered to
    the next. The returned list keeps the original request order, and the
    caller's inventory is never modified.
    """
    pending = [
        (index, request)
        for index, request in enumerate(requests)
        if _normalize(request.get("Status", "")) == "pending"
    ]

    pending.sort(
        key=lambda pair: (
            PRIORITY_ORDER.get(
                _normalize(pair[1].get("Priority", "")), DEFAULT_PRIORITY_RANK
            ),
            pair[0],
        )
    )

    # Work on copies so the original inventory stays untouched
    working_inventory = [dict(item) for item in inventory]
    results_by_index = {}

    for index, request in pending:
        result = recommend_resources(request, working_inventory)
        _deduct_from_inventory(working_inventory, result["recommendations"])

        results_by_index[index] = {
            "Request ID": result["request_id"],
            "Resource": request.get("Resource", ""),
            "Request Location": request.get("Location", ""),
            "Requested Quantity": result["requested_quantity"],
            "Recommended Quantity": result["allocated_quantity"],
            "Remaining Quantity": result["remaining_quantity"],
            "Matching Status": result["status"],
            "Recommendations": result["recommendations"],
        }

    # Return in the original request order
    return [results_by_index[i] for i in sorted(results_by_index)]