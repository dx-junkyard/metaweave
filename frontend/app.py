"""MetaWeave v1 — Streamlit frontend.

All data access goes through the FastAPI backend (BACKEND_URL env var).
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="MetaWeave v1", layout="wide")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------

if "search_results" not in st.session_state:
    st.session_state.search_results: list[dict] = []
if "stored_papers" not in st.session_state:
    st.session_state.stored_papers: dict[str, str] = {}   # arxiv_id -> object_name
if "structures" not in st.session_state:
    st.session_state.structures: dict[str, dict] = {}     # arxiv_id -> structure dict


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_search(query: str, max_results: int) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_URL}/api/search",
        params={"query": query, "max_results": max_results},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def api_fetch(paper: dict) -> str:
    """POST /api/fetch and return the MinIO object name."""
    resp = requests.post(f"{BACKEND_URL}/api/fetch", json=paper, timeout=120)
    resp.raise_for_status()
    return resp.json()["object_name"]


def api_presigned_url(object_name: str) -> str:
    resp = requests.get(
        f"{BACKEND_URL}/api/presigned-url",
        params={"object_name": object_name},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["url"]


def api_list_papers() -> list[str]:
    resp = requests.get(f"{BACKEND_URL}/api/papers", timeout=15)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

page = st.sidebar.radio("Navigation", ["Harvester Dashboard", "Validation View"])

# =========================================================================
# Page A — Harvester Dashboard
# =========================================================================
if page == "Harvester Dashboard":
    st.header("Harvester Dashboard")

    with st.form("search_form"):
        query = st.text_input("arXiv search query", value="cat:cs.AI")
        max_results = st.slider("Max results", 5, 50, 10)
        submitted = st.form_submit_button("Search")

    if submitted and query:
        with st.spinner("Querying arXiv via backend …"):
            try:
                st.session_state.search_results = api_search(query, max_results)
            except Exception as exc:
                st.error(f"Search failed: {exc}")

    results = st.session_state.search_results
    if not results:
        st.info("Enter a query and press **Search** to fetch papers from arXiv.")
    else:
        for idx, meta in enumerate(results):
            with st.container():
                cols = st.columns([5, 2, 1])
                authors = meta.get("authors", [])
                categories = meta.get("categories", [])

                with cols[0]:
                    st.markdown(f"**{meta['title']}**")
                    st.caption(
                        f"Authors: {', '.join(authors[:3])}{'…' if len(authors) > 3 else ''}  \n"
                        f"Categories: {', '.join(categories)}  |  "
                        f"License: {meta.get('license') or 'N/A'}"
                    )
                with cols[1]:
                    if meta.get("commercial_flag"):
                        st.warning("⚠ Commercial publisher suspected")
                    stored = st.session_state.stored_papers.get(meta["arxiv_id"])
                    if stored:
                        st.success(f"Stored: {stored}")
                with cols[2]:
                    if st.button("Fetch & Store", key=f"fetch_{idx}"):
                        with st.spinner("Downloading & storing PDF …"):
                            try:
                                object_name = api_fetch(meta)
                                st.session_state.stored_papers[meta["arxiv_id"]] = object_name
                                st.session_state.structures[meta["arxiv_id"]] = _mock_structure(meta)
                                st.rerun()
                            except Exception as exc:
                                st.error(f"Failed: {exc}")
                st.divider()

# =========================================================================
# Page B — Validation View
# =========================================================================
elif page == "Validation View":
    st.header("Structure Validation")

    # Merge locally stored papers with any already present in MinIO
    stored = dict(st.session_state.stored_papers)

    # Attempt to enrich from backend listing (best-effort)
    try:
        remote = api_list_papers()
        for obj in remote:
            # Derive a stable key from the object name
            key = obj.replace("arxiv/", "").replace(".pdf", "").replace("/", "_")
            stored.setdefault(key, obj)
    except Exception:
        pass

    if not stored:
        st.info("No papers stored yet. Use the **Harvester Dashboard** to fetch papers first.")
    else:
        paper_ids = list(stored.keys())
        selected_id = st.selectbox("Select paper", paper_ids)
        object_name = stored[selected_id]

        left, right = st.columns(2)

        # ----- Left pane: PDF viewer -----
        with left:
            st.subheader("PDF Viewer")
            try:
                url = api_presigned_url(object_name)
                components.iframe(url, height=700, scrolling=True)
            except Exception as exc:
                st.error(f"Could not load PDF: {exc}")

        # ----- Right pane: structure editor -----
        with right:
            st.subheader("Extracted Structure")

            s = st.session_state.structures.get(selected_id, _empty_structure(selected_id))

            with st.form("structure_form"):
                st.markdown("##### Problem Statement")
                bg = st.text_area("Background", value=s["problem"]["background"], height=80)
                prob = st.text_area("Problem", value=s["problem"]["problem"], height=80)

                st.markdown("##### Hypothesis")
                hyp_stmt = st.text_area("Statement", value=s["hypothesis"]["statement"], height=60)
                hyp_rat = st.text_area("Rationale", value=s["hypothesis"]["rationale"], height=60)

                st.markdown("##### Methodology")
                approach = st.text_area("Approach", value=s["methodology"]["approach"], height=60)
                techniques = st.text_area(
                    "Techniques (one per line)",
                    value="\n".join(s["methodology"]["techniques"]),
                    height=60,
                )

                st.markdown("##### Constraints")
                assumptions = st.text_area(
                    "Assumptions (one per line)",
                    value="\n".join(s["constraints"]["assumptions"]),
                    height=60,
                )
                limitations = st.text_area(
                    "Limitations (one per line)",
                    value="\n".join(s["constraints"]["limitations"]),
                    height=60,
                )

                st.markdown("##### Abstract Structure")
                variables = st.text_area(
                    "Variables (one per line)",
                    value="\n".join(s["abstract_structure"]["variables"]),
                    height=60,
                )
                edges_json = st.text_area(
                    "Edges (JSON list)",
                    value=json.dumps(s["abstract_structure"]["edges"], indent=2),
                    height=100,
                )

                st.markdown("##### Reviewer Notes")
                notes = st.text_area("Notes", value=s.get("reviewer_notes", ""), height=60)

                save = st.form_submit_button("Save edits")

            if save:
                try:
                    edges_parsed = json.loads(edges_json)
                except Exception:
                    edges_parsed = s["abstract_structure"]["edges"]
                    st.warning("Could not parse edges JSON — kept previous value.")

                s = {
                    "paper_id": selected_id,
                    "problem": {"background": bg, "problem": prob},
                    "hypothesis": {"statement": hyp_stmt, "rationale": hyp_rat},
                    "methodology": {
                        "approach": approach,
                        "techniques": [t.strip() for t in techniques.splitlines() if t.strip()],
                    },
                    "constraints": {
                        "assumptions": [a.strip() for a in assumptions.splitlines() if a.strip()],
                        "limitations": [l.strip() for l in limitations.splitlines() if l.strip()],
                    },
                    "abstract_structure": {
                        "variables": [v.strip() for v in variables.splitlines() if v.strip()],
                        "edges": edges_parsed,
                    },
                    "review_status": s.get("review_status", "pending"),
                    "reviewer_notes": notes,
                }
                st.session_state.structures[selected_id] = s
                st.success("Edits saved.")

            # Action buttons
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("✅ Approve", use_container_width=True):
                    s["review_status"] = "approved"
                    st.session_state.structures[selected_id] = s
                    st.success("Paper **approved**.")
            with btn_cols[1]:
                if st.button("❌ Reject", use_container_width=True):
                    s["review_status"] = "rejected"
                    st.session_state.structures[selected_id] = s
                    st.error("Paper **rejected**.")
            with btn_cols[2]:
                if st.button("🔄 Re-Extract", use_container_width=True):
                    s["review_status"] = "pending"
                    st.session_state.structures[selected_id] = s
                    st.info("Marked for **re-extraction**.")

            st.caption(f"Current status: **{s.get('review_status', 'pending')}**")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_structure(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "problem": {"background": "", "problem": ""},
        "hypothesis": {"statement": "", "rationale": ""},
        "methodology": {"approach": "", "techniques": []},
        "constraints": {"assumptions": [], "limitations": []},
        "abstract_structure": {
            "variables": ["Variable_A", "Variable_B"],
            "edges": [{"source": "Variable_A", "target": "Variable_B", "relation": "causes"}],
        },
        "review_status": "pending",
        "reviewer_notes": "",
    }


def _mock_structure(meta: dict) -> dict:
    s = _empty_structure(meta["arxiv_id"])
    s["problem"]["background"] = "(Auto-extracted from abstract) " + meta.get("summary", "")[:200]
    s["problem"]["problem"] = "To be extracted by LLM."
    s["hypothesis"]["statement"] = "To be extracted by LLM."
    return s
