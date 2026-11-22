from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from smart_priority import (
    assess_request,
    classify_batch,
    ml_status,
    pretty_category,
)
from resource_matcher import match_all_pending_requests

st.set_page_config(
    page_title="ReliefMatch",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REQUEST_COLUMNS = [
    "ID", "Emergency", "Location", "Description",
    "Resource", "Quantity", "Priority", "Status", "AI Category",
]
INVENTORY_COLUMNS = ["Resource", "Location", "Available"]

EMERGENCY_TYPES = [
    "Flood", "Earthquake", "Cyclone", "Fire",
    "Medical Emergency", "Food Shortage", "Other",
]
RESOURCE_TYPES = [
    "Drinking Water", "Food Packets", "First Aid Kits",
    "Medicines", "Rescue Boats", "Rescue Equipment",
]

PRIORITY_LEVELS = ["Critical", "High", "Medium", "Low"]
PRIORITY_COLORS = {
    "Critical": "#d62728",
    "High": "#ff7f0e",
    "Medium": "#e0b000",
    "Low": "#2ca02c",
}
PRIORITY_ICONS = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
STATUS_ICONS = {"Pending": "⏳", "Allocated": "✅"}

LOW_STOCK_LIMIT = 50

REQUEST_EXAMPLES = [
    {
        "label": "🌊 Family trapped",
        "emergency": "Flood",
        "location": "Meerut",
        "description": "Family with two children trapped on the roof, water rising fast.",
        "resource": "Rescue Boats",
        "quantity": 2,
    },
    {
        "label": "💧 Water shortage",
        "emergency": "Flood",
        "location": "Ghaziabad",
        "description": "Water supply cut after flooding, about 200 people need clean drinking water.",
        "resource": "Drinking Water",
        "quantity": 200,
    },
    {
        "label": "🩹 Medical supplies",
        "emergency": "Medical Emergency",
        "location": "Noida",
        "description": "Injured people at a relief camp need first aid, elderly patients present.",
        "resource": "First Aid Kits",
        "quantity": 30,
    },
]

SUGGESTED_QUESTIONS = [
    "What precautions should people take during a flood?",
    "How should families prepare an emergency kit?",
    "What should be done after a flood to stay safe?",
]


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.25);
        padding: 12px 16px;
        border-radius: 12px;
    }
    .pill {
        display: inline-block;
        padding: 2px 12px;
        border-radius: 999px;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers that work on older and newer Streamlit versions
# ---------------------------------------------------------------------------
def _stretch(func, *args, **kwargs):
    """Call a Streamlit element with full width, whichever API the version has."""
    try:
        return func(*args, width="stretch", **kwargs)
    except Exception:
        return func(*args, use_container_width=True, **kwargs)


def show_df(df, **kwargs):
    return _stretch(st.dataframe, df, hide_index=True, **kwargs)


def show_chart(chart):
    return _stretch(st.altair_chart, chart)


def stretch_button(label, **kwargs):
    return _stretch(st.button, label, **kwargs)


def stretch_submit(label, **kwargs):
    return _stretch(st.form_submit_button, label, **kwargs)


def pill(priority):
    color = PRIORITY_COLORS.get(priority, "#7f7f7f")
    return f'<span class="pill" style="background:{color}">{priority}</span>'


def normalize(value):
    return str(value if value is not None else "").strip().casefold()


# ---------------------------------------------------------------------------
# Data and state
# ---------------------------------------------------------------------------
def default_requests():
    return [
        {"ID": "REQ-001", "Emergency": "Flood", "Location": "Meerut",
         "Description": "Villages cut off by flooding.",
         "Resource": "Drinking Water", "Quantity": 100,
         "Priority": "High", "Status": "Pending"},
        {"ID": "REQ-002", "Emergency": "Medical Emergency", "Location": "Ghaziabad",
         "Description": "Shortage of basic medical supplies.",
         "Resource": "First Aid Kits", "Quantity": 20,
         "Priority": "Medium", "Status": "Pending"},
        {"ID": "REQ-003", "Emergency": "Food Shortage", "Location": "Noida",
         "Description": "Families in need of food.",
         "Resource": "Food Packets", "Quantity": 150,
         "Priority": "High", "Status": "Allocated"},
        {"ID": "REQ-004", "Emergency": "Cyclone", "Location": "Haridwar",
         "Description": "Villages cut off, families stranded and need boats.",
         "Resource": "Rescue Boats", "Quantity": 4,
         "Priority": "Critical", "Status": "Pending"},
        {"ID": "REQ-005", "Emergency": "Flood", "Location": "Meerut",
         "Description": "Drinking water contaminated after flooding.",
         "Resource": "Drinking Water", "Quantity": 300,
         "Priority": "Medium", "Status": "Pending"},
    ]


def default_inventory():
    return [
        {"Resource": "Drinking Water", "Location": "Meerut", "Available": 500},
        {"Resource": "Food Packets", "Location": "Ghaziabad", "Available": 300},
        {"Resource": "First Aid Kits", "Location": "Noida", "Available": 100},
        {"Resource": "Rescue Boats", "Location": "Haridwar", "Available": 10},
    ]


def init_state():
    if "requests" not in st.session_state:
        st.session_state["requests"] = default_requests()
    if "inventory" not in st.session_state:
        st.session_state["inventory"] = default_inventory()
    st.session_state.setdefault("form_version", 0)
    st.session_state.setdefault("flash", None)
    st.session_state.setdefault("toast", None)
    st.session_state.setdefault("log", [])
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("queued_question", None)


def reset_demo_data():
    st.session_state["requests"] = default_requests()
    st.session_state["inventory"] = default_inventory()
    st.session_state["log"] = []
    st.session_state["form_version"] += 1
    st.session_state["toast"] = ("Demo data has been reset.", "🔄")


def add_log(text):
    st.session_state["log"].insert(
        0, {"Time": datetime.now().strftime("%H:%M:%S"), "Event": text}
    )


def requests_df():
    return pd.DataFrame(st.session_state["requests"], columns=REQUEST_COLUMNS)


def inventory_df():
    return pd.DataFrame(st.session_state["inventory"], columns=INVENTORY_COLUMNS)


def next_request_id():
    numbers = []
    for r in st.session_state["requests"]:
        try:
            numbers.append(int(str(r["ID"]).split("-")[1]))
        except (IndexError, ValueError):
            pass
    return f"REQ-{(max(numbers) if numbers else 0) + 1:03d}"


def display_requests(df):
    """Add colour icons so priority and status are easy to scan."""
    out = df.copy()
    out["Priority"] = out["Priority"].map(
        lambda p: f"{PRIORITY_ICONS.get(p, '⚪')} {p}"
    )
    out["Status"] = out["Status"].map(lambda s: f"{STATUS_ICONS.get(s, '•')} {s}")
    out["AI Category"] = out["AI Category"].fillna("")
    return out


@st.cache_resource(show_spinner="Loading the knowledge base (the first run can take a few minutes)...")
def get_rag_engine():
    """Create the RAG engine once and reuse it. Imported here so the other
    pages still work even if the AI libraries are not installed."""
    from rag.rag_engine import RAGEngine

    return RAGEngine().prepare()


# ---------------------------------------------------------------------------
# Allocation logic (uses the recommendations from resource_matcher)
# ---------------------------------------------------------------------------
def approve_allocation(result):
    """Deduct the recommended stock and update the request. Returns True on success."""
    allocated = int(result["Recommended Quantity"])
    if allocated <= 0:
        return False

    for rec in result["Recommendations"]:
        to_deduct = int(rec["Recommended Quantity"])
        for row in st.session_state["inventory"]:
            if to_deduct <= 0:
                break
            if (
                normalize(row["Resource"]) == normalize(rec["Resource"])
                and normalize(row["Location"]) == normalize(rec["Location"])
            ):
                used = min(int(row["Available"]), to_deduct)
                row["Available"] = int(row["Available"]) - used
                to_deduct -= used

    remaining = int(result["Remaining Quantity"])
    for req in st.session_state["requests"]:
        if req["ID"] == result["Request ID"]:
            if remaining <= 0:
                req["Status"] = "Allocated"
            else:
                req["Quantity"] = remaining  # the rest stays pending
            break

    sources = ", ".join(
        f"{r['Recommended Quantity']} from {r['Location']}"
        for r in result["Recommendations"]
    )
    if remaining <= 0:
        add_log(f"{result['Request ID']}: fully allocated ({sources}).")
    else:
        add_log(
            f"{result['Request ID']}: partially allocated {allocated} units ({sources}); "
            f"{remaining} still needed."
        )
    return True


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def priority_chart(req_df):
    pending = req_df[req_df["Status"] == "Pending"]
    counts = pending["Priority"].value_counts()
    data = pd.DataFrame(
        {
            "Priority": PRIORITY_LEVELS,
            "Requests": [int(counts.get(level, 0)) for level in PRIORITY_LEVELS],
        }
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("Priority:N", sort=PRIORITY_LEVELS, title=None),
            y=alt.Y("Requests:Q", title="Pending requests", axis=alt.Axis(tickMinStep=1)),
            color=alt.Color(
                "Priority:N",
                scale=alt.Scale(
                    domain=PRIORITY_LEVELS,
                    range=[PRIORITY_COLORS[p] for p in PRIORITY_LEVELS],
                ),
                legend=None,
            ),
            tooltip=["Priority", "Requests"],
        )
        .properties(height=280)
    )
    show_chart(chart)


def demand_supply_chart(req_df, inv_df):
    pending = req_df[req_df["Status"] == "Pending"]
    demand = pending.groupby("Resource")["Quantity"].sum()
    supply = inv_df.groupby("Resource")["Available"].sum()
    data = pd.DataFrame({"Needed": demand, "In stock": supply}).fillna(0)
    data.index.name = "Resource"
    data = data.reset_index()

    if data.empty:
        st.info("Nothing to compare yet.")
        return

    long_df = data.melt(id_vars="Resource", var_name="Type", value_name="Units")
    try:
        chart = (
            alt.Chart(long_df)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("Resource:N", title=None),
                xOffset="Type:N",
                y=alt.Y("Units:Q", title="Units"),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=["Needed", "In stock"], range=["#d62728", "#1f77b4"]
                    ),
                ),
                tooltip=["Resource", "Type", "Units"],
            )
            .properties(height=280)
        )
        show_chart(chart)
    except Exception:
        st.bar_chart(data.set_index("Resource"))


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_dashboard():
    st.title("📊 ReliefMatch Dashboard")
    st.caption("Live overview of emergency requests, stock and allocation activity.")

    req_df = requests_df()
    inv_df = inventory_df()
    pending = req_df[req_df["Status"] == "Pending"]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total requests", len(req_df))
    c2.metric("Pending", len(pending))
    c3.metric("Allocated", int((req_df["Status"] == "Allocated").sum()))
    c4.metric("Critical pending", int((pending["Priority"] == "Critical").sum()))
    c5.metric("Units in stock", int(inv_df["Available"].sum()))

    low = inv_df[inv_df["Available"] < LOW_STOCK_LIMIT]
    if not low.empty:
        items = ", ".join(f"{r.Resource} in {r.Location} ({r.Available})" for r in low.itertuples())
        st.warning(f"Low stock (under {LOW_STOCK_LIMIT} units): {items}")

    tab_overview, tab_requests, tab_activity = st.tabs(
        ["📈 Overview", "📋 Requests", "🕒 Activity log"]
    )

    with tab_overview:
        left, right = st.columns(2)
        with left:
            st.subheader("Pending requests by priority")
            priority_chart(req_df)
        with right:
            st.subheader("Needed vs in stock")
            demand_supply_chart(req_df, inv_df)

    with tab_requests:
        f1, f2, f3 = st.columns([2, 2, 3])
        status_filter = f1.multiselect(
            "Status", ["Pending", "Allocated"], default=["Pending", "Allocated"], key="dash_status"
        )
        priority_filter = f2.multiselect(
            "Priority", PRIORITY_LEVELS, default=PRIORITY_LEVELS, key="dash_priority"
        )
        search = f3.text_input("Search", placeholder="Location, resource or description", key="dash_search")

        view = req_df[req_df["Status"].isin(status_filter) & req_df["Priority"].isin(priority_filter)]
        if search.strip():
            term = search.strip()
            mask = (
                view["Location"].str.contains(term, case=False, na=False, regex=False)
                | view["Resource"].str.contains(term, case=False, na=False, regex=False)
                | view["Description"].str.contains(term, case=False, na=False, regex=False)
            )
            view = view[mask]

        if view.empty:
            st.info("No requests match these filters.")
        else:
            show_df(display_requests(view))
            st.caption(f"Showing {len(view)} of {len(req_df)} requests.")

    with tab_activity:
        if st.session_state["log"]:
            show_df(pd.DataFrame(st.session_state["log"]))
        else:
            st.info("No allocations approved yet. Approve one on the Resource Allocation page.")

    st.caption("All figures are sample data for demonstration.")


def fill_example(version, example):
    """Button callback: fill the request form with an example."""
    st.session_state[f"emergency_{version}"] = example["emergency"]
    st.session_state[f"location_{version}"] = example["location"]
    st.session_state[f"description_{version}"] = example["description"]
    st.session_state[f"resource_{version}"] = example["resource"]
    st.session_state[f"quantity_{version}"] = example["quantity"]


def page_requests():
    st.title("🆘 Emergency Requests")
    st.caption("Register a request. Its priority is suggested automatically as you type.")

    tab_new, tab_batch, tab_all = st.tabs(
        ["➕ New request", "🤖 UiPath / Batch Import", "📋 All requests"]
    )

    with tab_new:
        if st.session_state["flash"]:
            st.success(st.session_state["flash"])
            st.session_state["flash"] = None

        v = st.session_state["form_version"]

        st.write("**Quick examples** (click to fill the form):")
        example_cols = st.columns(len(REQUEST_EXAMPLES))
        for col, example in zip(example_cols, REQUEST_EXAMPLES):
            with col:
                stretch_button(
                    example["label"],
                    key=f"example_{example['label']}_{v}",
                    on_click=fill_example,
                    args=(v, example),
                )

        a, b = st.columns(2)
        emergency = a.selectbox("Emergency type", EMERGENCY_TYPES, key=f"emergency_{v}")
        location = b.text_input("Location", key=f"location_{v}", placeholder="e.g. Meerut")
        description = st.text_area(
            "Emergency description",
            key=f"description_{v}",
            placeholder="Describe the situation: who is affected, how urgent it is...",
        )
        c, d = st.columns(2)
        resource = c.selectbox("Required resource", RESOURCE_TYPES, key=f"resource_{v}")
        quantity = d.number_input(
            "Required quantity", min_value=1, max_value=100000, value=10, step=1, key=f"quantity_{v}"
        )

        assessment = assess_request(emergency, description, resource, int(quantity))
        priority = assessment["priority"]
        ml = assessment["ml"]
        with st.container(border=True):
            st.markdown(f"**Suggested priority:** {pill(priority)}", unsafe_allow_html=True)
            st.caption(" • ".join(assessment["reasons"]))
            if ml:
                st.markdown(
                    f"🧠 **AI category:** {ml['label']} ({ml['confidence']:.0%} confidence)"
                )
                if ml["requires_review"]:
                    st.warning(
                        "Low AI confidence: a person should review this request. "
                        "The AI did not change the priority."
                    )
                elif assessment["ml_applied"]:
                    st.caption("The AI result raised the priority above the rule-based level.")
                st.caption("AI suggestions are not verified facts. Check the original message.")
            elif assessment["ml_note"]:
                st.caption(f"🧠 {assessment['ml_note']}")
            st.caption("The suggestion updates after you leave the description box.")

        if stretch_button("Submit emergency request", type="primary", key=f"submit_request_{v}"):
            if not location.strip() or not description.strip():
                st.error("Please enter both the location and the emergency description.")
            else:
                new_id = next_request_id()
                st.session_state["requests"].append({
                    "ID": new_id,
                    "Emergency": emergency,
                    "Location": location.strip().title(),
                    "Description": description.strip(),
                    "Resource": resource,
                    "Quantity": int(quantity),
                    "Priority": priority,
                    "Status": "Pending",
                    "AI Category": (
                        ("⚠️ " if ml["requires_review"] else "")
                        + f"{ml['label']} ({ml['confidence']:.0%})"
                        if ml else ""
                    ),
                })
                add_log(f"{new_id}: new {priority} request for {int(quantity)} × {resource}.")
                st.session_state["flash"] = (
                    f"Request {new_id} submitted. Assigned priority: {priority}."
                )
                st.session_state["toast"] = (f"{new_id} submitted ({priority})", "✅")
                st.session_state["form_version"] += 1
                st.rerun()

    with tab_batch:
        st.write("### 🤖 UiPath RPA & Batch Request Ingestion")
        st.caption(
            "UiPath robots can monitor emergency helplines, emails, or field spreadsheets and feed CSV files here for automatic AI analysis and queueing."
        )

        c1, c2 = st.columns([2, 1])
        with c1:
            uploaded_file = st.file_uploader(
                "Upload distress CSV (from UiPath or field team)",
                type=["csv"],
                key="uipath_csv_upload",
            )
        with c2:
            st.write("**Or load local file:**")
            local_sample = Path("uipath_incoming_emergencies.csv")
            has_sample = local_sample.exists()
            if stretch_button(
                "📂 Load uipath_incoming_emergencies.csv",
                disabled=not has_sample,
                key="btn_load_uipath_sample",
            ):
                st.session_state["selected_batch_file"] = "local"

        file_to_process = None
        if uploaded_file is not None:
            file_to_process = uploaded_file
        elif st.session_state.get("selected_batch_file") == "local" and local_sample.exists():
            file_to_process = local_sample

        if file_to_process is not None:
            try:
                df_in = pd.read_csv(file_to_process)
                required_cols = {"Emergency", "Location", "Description", "Resource", "Quantity"}
                if not required_cols.issubset(set(df_in.columns)):
                    st.error(f"Missing required columns. Expected: {', '.join(required_cols)}")
                else:
                    st.write(f"**Found {len(df_in)} request(s) to import:**")
                    st.dataframe(df_in, use_container_width=True)

                    if stretch_button(
                        f"⚡ Ingest {len(df_in)} requests with AI Priority Assessment",
                        type="primary",
                        key="btn_confirm_ingest",
                    ):
                        imported_count = 0
                        for _, row in df_in.iterrows():
                            em = str(row.get("Emergency", "Other")).strip()
                            loc = str(row.get("Location", "")).strip().title()
                            desc = str(row.get("Description", "")).strip()
                            res = str(row.get("Resource", "Other")).strip()
                            try:
                                qty = int(row.get("Quantity", 1))
                            except (ValueError, TypeError):
                                qty = 1

                            if not loc or not desc:
                                continue

                            assessment = assess_request(em, desc, res, qty)
                            pri = assessment["priority"]
                            ml = assessment["ml"]
                            n_id = next_request_id()

                            st.session_state["requests"].append({
                                "ID": n_id,
                                "Emergency": em,
                                "Location": loc,
                                "Description": desc,
                                "Resource": res,
                                "Quantity": qty,
                                "Priority": pri,
                                "Status": "Pending",
                                "AI Category": (
                                    ("⚠️ " if ml and ml["requires_review"] else "")
                                    + f"{ml['label']} ({ml['confidence']:.0%})"
                                    if ml else ""
                                ),
                            })
                            imported_count += 1

                        add_log(f"UiPath RPA: imported {imported_count} requests with AI assessment.")
                        st.session_state["toast"] = (
                            f"Successfully imported {imported_count} requests!",
                            "🤖",
                        )
                        st.session_state.pop("selected_batch_file", None)
                        st.session_state["form_version"] += 1
                        st.rerun()
            except Exception as ex:
                st.error(f"Error reading CSV: {ex}")

    with tab_all:
        req_df = requests_df()
        f1, f2 = st.columns(2)
        status_filter = f1.multiselect(
            "Status", ["Pending", "Allocated"], default=["Pending", "Allocated"], key="all_status"
        )
        priority_filter = f2.multiselect(
            "Priority", PRIORITY_LEVELS, default=PRIORITY_LEVELS, key="all_priority"
        )
        view = req_df[req_df["Status"].isin(status_filter) & req_df["Priority"].isin(priority_filter)]

        if view.empty:
            st.info("No requests match these filters.")
        else:
            show_df(display_requests(view))

        down_col, del_fulfilled_col = st.columns([1, 1])
        with down_col:
            st.download_button(
                "⬇️ Download requests as CSV",
                req_df.to_csv(index=False).encode("utf-8"),
                file_name="reliefmatch_requests.csv",
                mime="text/csv",
            )

        allocated_count = sum(
            1 for r in st.session_state["requests"] if r.get("Status") == "Allocated"
        )
        with del_fulfilled_col:
            if allocated_count > 0:
                if stretch_button(
                    f"🧹 Remove all fulfilled requests ({allocated_count})",
                    key="remove_fulfilled_btn",
                ):
                    st.session_state["requests"] = [
                        r
                        for r in st.session_state["requests"]
                        if r.get("Status") != "Allocated"
                    ]
                    add_log(f"Removed {allocated_count} fulfilled request(s).")
                    st.session_state["toast"] = (
                        f"Removed {allocated_count} fulfilled request(s).",
                        "🧹",
                    )
                    st.rerun()

        st.divider()
        st.write("##### 🗑️ Delete an unwanted / mistaken request")
        if st.session_state["requests"]:
            req_options = {
                r["ID"]: (
                    f"{r['ID']} · {r['Emergency']} ({r['Location']}) · "
                    f"{r['Quantity']} × {r['Resource']} [{r.get('Status', 'Pending')}]"
                )
                for r in st.session_state["requests"]
            }
            sel_col, btn_col = st.columns([3, 1])
            selected_to_delete = sel_col.selectbox(
                "Select request to remove",
                options=list(req_options.keys()),
                format_func=lambda req_id: req_options[req_id],
                key="select_req_to_delete",
                label_visibility="collapsed",
            )
            with btn_col:
                if stretch_button("🗑️ Delete request", key="btn_delete_req"):
                    st.session_state["requests"] = [
                        r
                        for r in st.session_state["requests"]
                        if r["ID"] != selected_to_delete
                    ]
                    add_log(f"{selected_to_delete}: request deleted by user.")
                    st.session_state["toast"] = (
                        f"Request {selected_to_delete} deleted.",
                        "🗑️",
                    )
                    st.rerun()


def page_inventory():
    st.title("📦 Resource Inventory")
    st.caption("Check stock levels and update the inventory.")

    tab_stock, tab_manage = st.tabs(["📦 Stock levels", "✏️ Manage inventory"])

    with tab_stock:
        inv_df = inventory_df()
        threshold = st.slider("Low-stock alert threshold", 0, 500, LOW_STOCK_LIMIT, step=10)
        search = st.text_input("Search by resource or location", key="inv_search")

        view = inv_df
        if search.strip():
            term = search.strip()
            mask = (
                view["Resource"].str.contains(term, case=False, na=False, regex=False)
                | view["Location"].str.contains(term, case=False, na=False, regex=False)
            )
            view = view[mask]

        if view.empty:
            st.info("No matching inventory found.")
        else:
            show_df(
                view,
                column_config={
                    "Available": st.column_config.ProgressColumn(
                        "Available",
                        min_value=0,
                        max_value=int(max(inv_df["Available"].max(), 1)),
                        format="%d",
                    )
                },
            )

        low = inv_df[inv_df["Available"] < threshold]
        if not low.empty:
            st.warning(
                "Below the alert threshold: "
                + ", ".join(f"{r.Resource} in {r.Location} ({r.Available})" for r in low.itertuples())
            )

        totals = inv_df.groupby("Resource", as_index=False)["Available"].sum()
        if not totals.empty:
            st.subheader("Total stock by resource")
            st.bar_chart(totals.set_index("Resource"))

    with tab_manage:
        st.write("Edit values, add rows with the **+** at the bottom, or delete rows, then save.")
        options = sorted(set(RESOURCE_TYPES) | set(inventory_df()["Resource"].dropna()))

        with st.form("inventory_form"):
            edited = _stretch(
                st.data_editor,
                inventory_df(),
                num_rows="dynamic",
                hide_index=True,
                column_config={
                    "Resource": st.column_config.SelectboxColumn("Resource", options=options, required=True),
                    "Location": st.column_config.TextColumn("Location", required=True),
                    "Available": st.column_config.NumberColumn(
                        "Available", min_value=0, step=1, required=True
                    ),
                },
                key="inventory_editor",
            )
            saved = stretch_submit("💾 Save inventory", type="primary")

        if saved:
            clean = []
            for _, row in edited.iterrows():
                resource = str(row["Resource"] or "").strip()
                location = str(row["Location"] or "").strip().title()
                if not resource or not location or pd.isna(row["Available"]):
                    continue
                clean.append(
                    {"Resource": resource, "Location": location, "Available": int(row["Available"])}
                )
            st.session_state["inventory"] = clean
            add_log("Inventory updated manually.")
            st.session_state["toast"] = ("Inventory saved.", "💾")
            st.rerun()

    st.caption("Inventory values are simulated.")


def page_allocation():
    st.title("🚚 Resource Allocation")
    st.caption("Review recommended sources, then approve allocations to update stock.")

    results = match_all_pending_requests(
        st.session_state["requests"], st.session_state["inventory"]
    )

    if not results:
        st.info("There are no pending emergency requests to match.")
        return

    fully = [r for r in results if r["Matching Status"] == "Fully Matchable"]
    partial = [r for r in results if r["Matching Status"] == "Partially Matchable"]
    unavailable = [r for r in results if r["Matching Status"] == "Unavailable"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pending requests", len(results))
    c2.metric("Fully matchable", len(fully))
    c3.metric("Partially matchable", len(partial))
    c4.metric("Unavailable", len(unavailable))

    priority_lookup = {r["ID"]: r["Priority"] for r in st.session_state["requests"]}

    top_left, top_right = st.columns([3, 2])
    with top_left:
        status_filter = st.radio(
            "Show",
            ["All", "Fully matchable", "Partially matchable", "Unavailable"],
            horizontal=True,
        )
    with top_right:
        if stretch_button(
            f"✅ Approve all fully matchable ({len(fully)})",
            type="primary",
            disabled=not fully,
            key="approve_all",
        ):
            approved = sum(1 for r in fully if approve_allocation(r))
            st.session_state["toast"] = (f"{approved} allocation(s) approved.", "✅")
            st.rerun()

    wanted = {
        "All": None,
        "Fully matchable": "Fully Matchable",
        "Partially matchable": "Partially Matchable",
        "Unavailable": "Unavailable",
    }[status_filter]

    shown = [r for r in results if wanted is None or r["Matching Status"] == wanted]
    if not shown:
        st.info("No requests in this category.")

    for result in shown:
        request_id = result["Request ID"]
        priority = priority_lookup.get(request_id, "")
        requested = max(int(result["Requested Quantity"]), 1)
        recommended = int(result["Recommended Quantity"])

        with st.container(border=True):
            head, action = st.columns([4, 1])
            with head:
                st.markdown(
                    f"**{request_id}** · {result['Resource']} · {result['Request Location']} "
                    f"&nbsp; {pill(priority) if priority else ''}",
                    unsafe_allow_html=True,
                )
                st.progress(
                    min(recommended / requested, 1.0),
                    text=f"{result['Matching Status']}: {recommended} of {requested} units available",
                )
            with action:
                if stretch_button(
                    "Approve",
                    key=f"approve_{request_id}",
                    disabled=recommended <= 0,
                    type="primary" if result["Matching Status"] == "Fully Matchable" else "secondary",
                ):
                    approve_allocation(result)
                    st.session_state["toast"] = (f"{request_id} allocation approved.", "✅")
                    st.rerun()

            if result["Recommendations"]:
                with st.expander("Recommended sources"):
                    show_df(pd.DataFrame(result["Recommendations"]))
            else:
                st.warning("No suitable inventory is currently available for this request.")

    st.info(
        "Approving deducts stock in this demo session. A partly filled request stays "
        "pending with the remaining quantity."
    )
    st.warning("These sample values must not be used for real emergency decisions.")


def page_assistant():
    st.title("📚 Disaster Knowledge Assistant")
    st.caption("Ask about disaster preparedness. Answers come from the PDFs in the documents folder.")

    history = st.session_state["chat_history"]

    top_left, top_right = st.columns([4, 1])
    with top_left:
        st.write("**Try asking:**")
    with top_right:
        if history and stretch_button("🗑️ Clear chat", key="clear_chat"):
            st.session_state["chat_history"] = []
            st.rerun()

    suggestion_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, suggestion in zip(suggestion_cols, SUGGESTED_QUESTIONS):
        with col:
            if stretch_button(suggestion, key=f"suggest_{suggestion}"):
                st.session_state["queued_question"] = suggestion
                st.rerun()

    question = st.chat_input("Ask about floods, safety precautions, relief...")
    if not question and st.session_state["queued_question"]:
        question = st.session_state["queued_question"]
    st.session_state["queued_question"] = None

    if question and question.strip():
        history.append({"role": "user", "content": question.strip()})
        try:
            engine = get_rag_engine()
            with st.spinner("Searching the documents..."):
                rag_result = engine.ask(question.strip())
            history.append({
                "role": "assistant",
                "content": rag_result["answer"],
                "sources": rag_result["sources"],
                "passages": rag_result["passages"],
                "notice": rag_result["notice"],
            })
        except ImportError as error:
            history.append({
                "role": "assistant",
                "error": f"A required package is missing: {error}. Run: pip install -r requirements.txt",
            })
        except FileNotFoundError as error:
            history.append({"role": "assistant", "error": f"Could not load the PDFs: {error}"})
        except Exception as error:
            history.append({"role": "assistant", "error": f"Something went wrong: {error}"})

    if not history:
        st.info("No questions yet. Click a suggestion above or type your own question below.")

    for message in history:
        with st.chat_message(message["role"]):
            if message.get("error"):
                st.error(message["error"])
                continue
            if message.get("notice"):
                st.warning(message["notice"])
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption(
                    "Sources: "
                    + "; ".join(f"{s['source']}, p. {s['page']}" for s in message["sources"])
                )
                with st.expander("Show retrieved passages"):
                    for passage in message.get("passages", []):
                        st.markdown(
                            f"**{passage['source']}, p. {passage['page']}** "
                            f"(match score {passage['score']:.2f})"
                        )
                        st.write(passage["text"])

    st.caption(
        "General guidance from official documents. In an emergency, follow local "
        "authorities and call 112."
    )


SAMPLE_TWEETS_PATH = Path(__file__).resolve().parent / "ml" / "data" / "humaid_sample.csv"
CLASSIFIER_EXAMPLES = [
    "We urgently need drinking water and food for 200 people at the school shelter",
    "Bridge collapsed and power lines are down in the village",
    "Praying for everyone affected by the cyclone",
]
MAX_ROWS = 2000


@st.cache_data(show_spinner="Classifying messages with the AI model...")
def classify_texts(texts):
    """Classify a tuple of messages once and remember the result."""
    return pd.DataFrame(classify_batch(list(texts)))


def pretty_label(value):
    return pretty_category(value) if isinstance(value, str) and value else ""


def confusion_chart(human, ai):
    table = pd.crosstab(human.rename("Human label"), ai.rename("AI category"))
    labels = sorted(set(table.index) | set(table.columns))
    table = table.reindex(index=labels, columns=labels, fill_value=0)
    long = table.stack().rename("Count").reset_index()
    try:
        base = alt.Chart(long).encode(
            x=alt.X("AI category:N", sort=labels, title="AI category"),
            y=alt.Y("Human label:N", sort=labels, title="Human label"),
        )
        heat = base.mark_rect().encode(
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues"), legend=None),
            tooltip=["Human label", "AI category", "Count"],
        )
        numbers = base.mark_text(fontSize=11).encode(
            text="Count:Q",
            color=alt.condition(alt.datum.Count > 15, alt.value("white"), alt.value("black")),
        )
        show_chart((heat + numbers).properties(height=420))
    except Exception:
        st.dataframe(table)


def page_classifier():
    st.title("🔎 Message Classifier")
    st.caption("Real disaster tweets (HumAID) sorted into humanitarian categories by the trained AI model.")

    ready, message = ml_status()
    if not ready:
        st.error(message)
        return

    tab_try, tab_sample = st.tabs(["✍️ Try a message", "📊 Real tweet sample"])

    with tab_try:
        st.write("**Examples** (click to fill):")
        example_cols = st.columns(len(CLASSIFIER_EXAMPLES))
        for i, (col, example) in enumerate(zip(example_cols, CLASSIFIER_EXAMPLES)):
            with col:
                stretch_button(
                    f"Example {i + 1}",
                    key=f"clf_example_{i}",
                    on_click=lambda text=example: st.session_state.update(clf_text=text),
                )

        text = st.text_area(
            "Message", key="clf_text", placeholder="Paste a tweet or any disaster-related message..."
        )

        if text.strip():
            result = classify_batch([text])[0]
            with st.container(border=True):
                st.markdown(f"### {result['label']}")
                st.progress(result["confidence"], text=f"Confidence: {result['confidence']:.0%}")
                if result["requires_review"]:
                    st.warning("Low confidence: a person should review this message.")
                else:
                    st.success("Confident prediction.")
                st.markdown(
                    f"Maps to priority: {pill(result['ml_priority'])}", unsafe_allow_html=True
                )
                st.caption(
                    "In the request form, the AI only raises a priority when its confidence "
                    "is at least 60%. It never lowers one."
                )
        else:
            st.info("Type or paste a message above to see how the AI classifies it.")

    with tab_sample:
        source = st.radio(
            "Data source",
            ["Bundled real tweet sample", "Upload a CSV"],
            horizontal=True,
            key="clf_source",
        )

        data = None
        if source == "Bundled real tweet sample":
            if SAMPLE_TWEETS_PATH.exists():
                data = pd.read_csv(SAMPLE_TWEETS_PATH)
            else:
                st.info(f"Sample file not found. Put humaid_sample.csv in: {SAMPLE_TWEETS_PATH.parent}")
        else:
            uploaded = st.file_uploader(
                "CSV with a tweet_text column (class_label and event are optional)", type="csv"
            )
            if uploaded is not None:
                try:
                    data = pd.read_csv(uploaded)
                except Exception as error:
                    st.error(f"Could not read the CSV: {error}")

        if data is None:
            return
        if "tweet_text" not in data.columns:
            st.error("The CSV needs a column named tweet_text.")
            return

        data = data.dropna(subset=["tweet_text"]).reset_index(drop=True)
        if len(data) > MAX_ROWS:
            st.warning(f"Only the first {MAX_ROWS} rows are classified.")
            data = data.head(MAX_ROWS)

        predictions = classify_texts(tuple(data["tweet_text"].astype(str)))
        has_labels = "class_label" in data.columns

        results = pd.DataFrame({
            "Tweet": data["tweet_text"].astype(str),
            "Event": data["event"] if "event" in data.columns else "",
            "Human label": data["class_label"].map(pretty_label) if has_labels else "",
            "AI category": predictions["label"],
            "Confidence": predictions["confidence"],
            "Needs review": predictions["requires_review"],
        })
        if has_labels:
            results["Correct"] = results["Human label"] == results["AI category"]

        confident = results[~results["Needs review"]]
        flagged = results[results["Needs review"]]

        cols = st.columns(5 if has_labels else 2)
        cols[0].metric("Tweets classified", len(results))
        cols[1].metric("Flagged for review", f"{results['Needs review'].mean():.0%}")
        if has_labels:
            cols[2].metric("Accuracy vs human labels", f"{results['Correct'].mean():.1%}")
            cols[3].metric(
                "Accuracy when confident",
                f"{confident['Correct'].mean():.1%}" if len(confident) else "n/a",
            )
            cols[4].metric(
                "Accuracy when flagged",
                f"{flagged['Correct'].mean():.1%}" if len(flagged) else "n/a",
            )
            st.caption(
                "Confident predictions are much more reliable than flagged ones. "
                "That is why low-confidence results are sent for human review."
            )

        left, right = st.columns(2)
        with left:
            if has_labels:
                st.subheader("Accuracy per category")
                per_class = (
                    results.groupby("Human label")["Correct"].mean().rename("Accuracy").reset_index()
                )
                show_chart(
                    alt.Chart(per_class)
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        y=alt.Y("Human label:N", sort="-x", title=None),
                        x=alt.X("Accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
                        tooltip=["Human label", alt.Tooltip("Accuracy:Q", format=".0%")],
                    )
                    .properties(height=320)
                )
            else:
                st.subheader("Predicted categories")
                counts = results["AI category"].value_counts().rename("Tweets")
                st.bar_chart(counts)
        with right:
            st.subheader("Confidence levels")
            bins = pd.cut(
                results["Confidence"],
                bins=[0, 0.4, 0.6, 0.8, 1.0],
                labels=["Under 40%", "40-60%", "60-80%", "80-100%"],
                include_lowest=True,
            ).value_counts().reindex(["Under 40%", "40-60%", "60-80%", "80-100%"]).rename("Tweets")
            st.bar_chart(bins)

        if has_labels:
            with st.expander("Confusion matrix (human label vs AI category)"):
                confusion_chart(results["Human label"], results["AI category"])

        st.subheader("Tweet-by-tweet results")
        f1, f2, f3 = st.columns(3)
        only_wrong = f1.checkbox("Only wrong predictions", disabled=not has_labels)
        only_flagged = f2.checkbox("Only flagged for review")
        events = sorted(e for e in results["Event"].unique() if e)
        chosen_events = f3.multiselect("Event", events) if events else []

        view = results
        if only_wrong:
            view = view[~view["Correct"]]
        if only_flagged:
            view = view[view["Needs review"]]
        if chosen_events:
            view = view[view["Event"].isin(chosen_events)]

        shown = view.copy()
        shown["Review"] = shown["Needs review"].map(lambda flag: "⚠️" if flag else "")
        columns = ["Tweet", "Human label", "AI category", "Confidence", "Review"]
        if has_labels:
            shown["Match"] = shown["Correct"].map(lambda ok: "✅" if ok else "❌")
            columns.append("Match")
        if events:
            columns.insert(1, "Event")
        if not has_labels:
            columns.remove("Human label")

        if shown.empty:
            st.info("No tweets match these filters.")
        else:
            show_df(
                shown[columns],
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )
            st.caption(f"Showing {len(shown)} of {len(results)} tweets.")

        st.download_button(
            "⬇️ Download results as CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="classified_messages.csv",
            mime="text/csv",
        )
        st.caption(
            "Tweets come from the HumAID dataset. Check its license before sharing them publicly."
        )


# ---------------------------------------------------------------------------
# App shell
# ---------------------------------------------------------------------------
init_state()

PAGES = {
    "📊 Dashboard": page_dashboard,
    "🆘 Emergency Requests": page_requests,
    "🔎 Message Classifier": page_classifier,
    "📦 Resource Inventory": page_inventory,
    "🚚 Resource Allocation": page_allocation,
    "📚 Knowledge Assistant": page_assistant,
}

with st.sidebar:
    st.title("🚨 ReliefMatch")
    st.caption("AI-Powered Disaster Relief Coordination")
    st.divider()

    page = st.radio("Navigation", list(PAGES.keys()), key="nav")

    st.divider()
    sidebar_df = requests_df()
    pending_now = sidebar_df[sidebar_df["Status"] == "Pending"]
    s1, s2 = st.columns(2)
    s1.metric("Pending", len(pending_now))
    s2.metric("Critical", int((pending_now["Priority"] == "Critical").sum()))

    st.divider()
    ai_ready, ai_message = ml_status()
    if ai_ready:
        st.caption("🧠 AI classifier: connected")
    else:
        st.caption("🧠 AI classifier: not connected (rules only)")
        st.caption(ai_message)
    st.info("Demo Mode: Using sample data.")
    if stretch_button("🔄 Reset demo data", key="reset_demo"):
        reset_demo_data()
        st.rerun()

if st.session_state["toast"]:
    message, icon = st.session_state["toast"]
    st.session_state["toast"] = None
    st.toast(message, icon=icon)

PAGES[page]()

st.divider()
st.caption("ReliefMatch | Academic Project Prototype")