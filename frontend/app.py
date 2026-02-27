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
if "paper_metadata" not in st.session_state:
    st.session_state.paper_metadata: dict[str, dict] = {}  # arxiv_id -> paper meta
if "active_paper_id" not in st.session_state:
    st.session_state.active_paper_id: str | None = None


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


def api_extract(object_name: str, arxiv_id: str) -> dict:
    """POST /api/extract and return the extracted PaperStructure as a dict.

    Reasoning モデルは応答に数十秒〜数分かかる場合があるため timeout を長めに設定。
    """
    resp = requests.post(
        f"{BACKEND_URL}/api/extract",
        json={"object_name": object_name, "arxiv_id": arxiv_id},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()


def api_get_extract_result(arxiv_id: str) -> dict:
    """GET /api/extract-result/{arxiv_id} — load a saved structure from MinIO."""
    resp = requests.get(
        f"{BACKEND_URL}/api/extract-result/{arxiv_id}",
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def api_update_extract_result(arxiv_id: str, structure: dict) -> None:
    """PUT /api/extract-result/{arxiv_id} — persist an edited structure to MinIO."""
    resp = requests.put(
        f"{BACKEND_URL}/api/extract-result/{arxiv_id}",
        json=structure,
        timeout=30,
    )
    resp.raise_for_status()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_structure(paper_id: str) -> dict:
    return {
        "paper_id": paper_id,
        "title": "",
        "problem": {"background": "", "problem": ""},
        "hypothesis": {"statement": "", "rationale": ""},
        "methodology": {"approach": "", "techniques": []},
        "constraints": {"assumptions": [], "limitations": []},
        "abstract_structure": {
            "variables": [],
            "edges": [],
        },
        "review_status": "pending",
        "reviewer_notes": "",
    }


_STATUS_ICON: dict[str, str] = {
    "pending": "🟡",
    "approved": "✅",
    "rejected": "❌",
}

_STATUS_LABEL: dict[str, str] = {
    "pending": "🟡 Pending",
    "approved": "✅ Approved",
    "rejected": "❌ Rejected",
}

# ---------------------------------------------------------------------------
# Sidebar — Navigation + Paper list (Validation View)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## MetaWeave v1")
    page = st.radio("Navigation", ["Harvester Dashboard", "Validation View"])

    if page == "Validation View":
        st.divider()

        # Build paper registry for sidebar list
        _stored_sidebar: dict[str, str] = dict(st.session_state.stored_papers)
        try:
            _remote_sidebar = api_list_papers()
            for _obj_s in _remote_sidebar:
                _safe_id_s = _obj_s.rsplit("/", 1)[-1].replace(".pdf", "")
                _stored_sidebar.setdefault(_safe_id_s, _obj_s)
        except Exception:
            pass

        st.markdown("### Papers")
        _search_text = st.text_input(
            "Search",
            placeholder="Filter by title or ID…",
            label_visibility="collapsed",
        )
        _status_filter = st.selectbox(
            "Status filter",
            ["All", "Pending", "Approved", "Rejected"],
            label_visibility="collapsed",
        )
        st.markdown("---")

        for _aid, _obj_s in _stored_sidebar.items():
            _s = st.session_state.structures.get(_aid, {})
            _status = _s.get("review_status", "pending")

            if _status_filter != "All" and _status != _status_filter.lower():
                continue

            _title: str = (
                st.session_state.paper_metadata.get(_aid, {}).get("title")
                or _s.get("title")
                or _aid
            )

            if _search_text and (
                _search_text.lower() not in _title.lower()
                and _search_text.lower() not in _aid.lower()
            ):
                continue

            _icon = _STATUS_ICON.get(_status, "🟡")
            _label = f"{_icon} {_title}" if len(_title) <= 34 else f"{_icon} {_title[:31]}…"
            _is_active = _aid == st.session_state.active_paper_id

            if st.button(
                _label,
                key=f"sel_{_aid}",
                use_container_width=True,
                type="primary" if _is_active else "secondary",
            ):
                st.session_state.active_paper_id = _aid
                if _aid not in st.session_state.structures:
                    with st.spinner("Loading…"):
                        try:
                            _result = api_get_extract_result(_aid)
                            st.session_state.structures[_aid] = _result
                        except Exception:
                            st.session_state.structures[_aid] = _empty_structure(_aid)
                st.rerun()

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
                        arxiv_id = meta["arxiv_id"]
                        try:
                            with st.spinner("Downloading & storing PDF …"):
                                object_name = api_fetch(meta)
                                st.session_state.stored_papers[arxiv_id] = object_name
                                st.session_state.paper_metadata[arxiv_id] = meta
                            with st.spinner(
                                "Extracting structure using LLM (this may take a minute) …"
                            ):
                                structure = api_extract(object_name, arxiv_id)
                                st.session_state.structures[arxiv_id] = structure
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")
                st.divider()

# =========================================================================
# Page B — Validation View
# =========================================================================
elif page == "Validation View":
    # Build unified paper registry: safe_id / arxiv_id -> object_name
    stored: dict[str, str] = dict(st.session_state.stored_papers)
    try:
        remote = api_list_papers()
        for obj in remote:
            safe_id = obj.rsplit("/", 1)[-1].replace(".pdf", "")
            stored.setdefault(safe_id, obj)
    except Exception:
        pass

    if not stored:
        st.info("No papers stored yet. Use the **Harvester Dashboard** to fetch papers first.")
    else:
        active_id = st.session_state.active_paper_id

        if not active_id or active_id not in stored:
            st.info("← Select a paper from the sidebar to view and edit its structure.")
        else:
            object_name = stored[active_id]
            s = st.session_state.structures.get(active_id, _empty_structure(active_id))
            status = s.get("review_status", "pending")
            title = (
                st.session_state.paper_metadata.get(active_id, {}).get("title")
                or s.get("title")
                or active_id
            )

            # ── Header: prominent title + arXiv ID ───────────────────────────
            st.title(title)
            st.caption(
                f"arXiv ID: `{active_id}`  |  Status: {_STATUS_LABEL.get(status, '🟡 Pending')}"
            )

            # ── Action buttons (right-aligned) ───────────────────────────────
            _, act1, act2, act3 = st.columns([6, 1, 1, 1])
            with act1:
                if st.button("✅ Approve", use_container_width=True, key="btn_approve"):
                    s["review_status"] = "approved"
                    st.session_state.structures[active_id] = s
                    try:
                        api_update_extract_result(active_id, s)
                    except Exception as exc:
                        st.warning(f"Backend sync failed: {exc}")
                    st.rerun()
            with act2:
                if st.button("❌ Reject", use_container_width=True, key="btn_reject"):
                    s["review_status"] = "rejected"
                    st.session_state.structures[active_id] = s
                    try:
                        api_update_extract_result(active_id, s)
                    except Exception as exc:
                        st.warning(f"Backend sync failed: {exc}")
                    st.rerun()
            with act3:
                if st.button("🔄 Re-Extract", use_container_width=True, key="btn_reextract"):
                    try:
                        with st.spinner("Re-extracting…"):
                            result = api_extract(object_name, active_id)
                            st.session_state.structures[active_id] = result
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Re-extraction failed: {exc}")

            st.divider()

            # ── Main 2-column layout [5, 5] ──────────────────────────────────
            pdf_col, edit_col = st.columns([5, 5])

            # ── PDF Preview ──────────────────────────────────────────────────
            with pdf_col:
                st.markdown("**PDF Preview**")
                try:
                    pdf_url = api_presigned_url(object_name)
                    st.markdown(
                        f'<iframe src="{pdf_url}" width="100%" height="800px"'
                        ' style="border:none; border-radius:4px;"></iframe>',
                        unsafe_allow_html=True,
                    )
                except Exception as exc:
                    st.error(f"Could not load PDF: {exc}")

            # ── Structure Editor ─────────────────────────────────────────────
            with edit_col:
                st.markdown("**Extracted Structure**")

                with st.form(f"structure_form_{active_id}"):
                    tab1, tab2, tab3 = st.tabs(
                        ["Problem / Hypothesis", "Method / Constraints", "Abstract / Notes"]
                    )

                    with tab1:
                        st.markdown("#### Problem Statement")
                        bg = st.text_area(
                            "Background",
                            value=s["problem"]["background"],
                            height=130,
                        )
                        prob = st.text_area(
                            "Problem",
                            value=s["problem"]["problem"],
                            height=130,
                        )
                        st.markdown("#### Hypothesis")
                        hyp_stmt = st.text_area(
                            "Statement",
                            value=s["hypothesis"]["statement"],
                            height=110,
                        )
                        hyp_rat = st.text_area(
                            "Rationale",
                            value=s["hypothesis"]["rationale"],
                            height=110,
                        )

                    with tab2:
                        st.markdown("#### Methodology")
                        approach = st.text_area(
                            "Approach",
                            value=s["methodology"]["approach"],
                            height=130,
                        )
                        techniques = st.text_area(
                            "Techniques (one per line)",
                            value="\n".join(s["methodology"]["techniques"]),
                            height=110,
                        )
                        st.markdown("#### Constraints")
                        assumptions = st.text_area(
                            "Assumptions (one per line)",
                            value="\n".join(s["constraints"]["assumptions"]),
                            height=110,
                        )
                        limitations = st.text_area(
                            "Limitations (one per line)",
                            value="\n".join(s["constraints"]["limitations"]),
                            height=110,
                        )

                    with tab3:
                        st.markdown("#### Abstract Structure")
                        variables = st.text_area(
                            "Variables (one per line)",
                            value="\n".join(s["abstract_structure"]["variables"]),
                            height=110,
                        )
                        edges_json = st.text_area(
                            "Edges (JSON list)",
                            value=json.dumps(s["abstract_structure"]["edges"], indent=2),
                            height=150,
                        )
                        st.markdown("#### Reviewer Notes")
                        notes = st.text_area(
                            "Notes",
                            value=s.get("reviewer_notes", ""),
                            height=110,
                        )

                    save = st.form_submit_button(
                        "💾 Save edits",
                        use_container_width=True,
                        type="primary",
                    )

                if save:
                    try:
                        edges_parsed = json.loads(edges_json)
                    except Exception:
                        edges_parsed = s["abstract_structure"]["edges"]
                        st.warning("Could not parse edges JSON — kept previous value.")

                    updated: dict = {
                        "paper_id": active_id,
                        "title": s.get("title", ""),
                        "problem": {"background": bg, "problem": prob},
                        "hypothesis": {"statement": hyp_stmt, "rationale": hyp_rat},
                        "methodology": {
                            "approach": approach,
                            "techniques": [
                                t.strip() for t in techniques.splitlines() if t.strip()
                            ],
                        },
                        "constraints": {
                            "assumptions": [
                                a.strip() for a in assumptions.splitlines() if a.strip()
                            ],
                            "limitations": [
                                l.strip() for l in limitations.splitlines() if l.strip()
                            ],
                        },
                        "abstract_structure": {
                            "variables": [
                                v.strip() for v in variables.splitlines() if v.strip()
                            ],
                            "edges": edges_parsed,
                        },
                        "review_status": s.get("review_status", "pending"),
                        "reviewer_notes": notes,
                    }
                    st.session_state.structures[active_id] = updated

                    # Sync to backend / MinIO
                    try:
                        api_update_extract_result(active_id, updated)
                        st.success("Edits saved and synced to MinIO.")
                    except Exception as exc:
                        st.success("Edits saved locally.")
                        st.warning(f"Backend sync failed: {exc}")
