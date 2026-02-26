"""MetaWeave v1 — Streamlit UI for harvesting and validating paper structures."""

from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

from metaweave.harvester import PaperMeta, fetch_and_store, search_arxiv
from metaweave.schema import (
    AbstractStructure,
    CausalEdge,
    Constraints,
    Hypothesis,
    Methodology,
    PaperStructure,
    ProblemStatement,
    ReviewStatus,
)
from metaweave.storage import StorageManager

st.set_page_config(page_title="MetaWeave v1", layout="wide")

# ---------------------------------------------------------------------------
# Session-state defaults
# ---------------------------------------------------------------------------
if "search_results" not in st.session_state:
    st.session_state.search_results: list[PaperMeta] = []
if "stored_papers" not in st.session_state:
    st.session_state.stored_papers: dict[str, str] = {}  # arxiv_id -> object_name
if "structures" not in st.session_state:
    st.session_state.structures: dict[str, PaperStructure] = {}  # arxiv_id -> structure


def _get_storage() -> StorageManager:
    """Return a cached StorageManager instance."""
    if "storage" not in st.session_state:
        st.session_state.storage = StorageManager()
    return st.session_state.storage


# =========================================================================
# Sidebar navigation
# =========================================================================
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
        with st.spinner("Querying arXiv …"):
            st.session_state.search_results = search_arxiv(query, max_results=max_results)

    results = st.session_state.search_results
    if not results:
        st.info("Enter a query and press **Search** to fetch papers from arXiv.")
    else:
        for idx, meta in enumerate(results):
            with st.container():
                cols = st.columns([5, 2, 1])
                with cols[0]:
                    st.markdown(f"**{meta.title}**")
                    st.caption(
                        f"Authors: {', '.join(meta.authors[:3])}{'…' if len(meta.authors) > 3 else ''}  \n"
                        f"Categories: {', '.join(meta.categories)}  |  License: {meta.license or 'N/A'}"
                    )
                with cols[1]:
                    if meta.commercial_flag:
                        st.warning("⚠ Commercial publisher suspected")
                    status = st.session_state.stored_papers.get(meta.arxiv_id)
                    if status:
                        st.success(f"Stored: {status}")
                with cols[2]:
                    if st.button("Fetch & Store", key=f"fetch_{idx}"):
                        try:
                            storage = _get_storage()
                            obj = fetch_and_store(meta, storage)
                            st.session_state.stored_papers[meta.arxiv_id] = obj
                            # Create mock structure for validation
                            st.session_state.structures[meta.arxiv_id] = _mock_structure(meta)
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Failed: {exc}")
                st.divider()

# =========================================================================
# Page B — Validation View (2-pane)
# =========================================================================
elif page == "Validation View":
    st.header("Structure Validation")

    stored = st.session_state.stored_papers
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
                storage = _get_storage()
                url = storage.presigned_url("raw-papers", object_name)
                components.iframe(url, height=700, scrolling=True)
            except Exception as exc:
                st.error(f"Could not load PDF: {exc}")

        # ----- Right pane: Structure editor -----
        with right:
            st.subheader("Extracted Structure")

            struct = st.session_state.structures.get(selected_id)
            if struct is None:
                struct = PaperStructure(paper_id=selected_id)
                st.session_state.structures[selected_id] = struct

            with st.form("structure_form"):
                st.markdown("##### Problem Statement")
                bg = st.text_area("Background", value=struct.problem.background, height=80)
                prob = st.text_area("Problem", value=struct.problem.problem, height=80)

                st.markdown("##### Hypothesis")
                hyp_stmt = st.text_area("Statement", value=struct.hypothesis.statement, height=60)
                hyp_rat = st.text_area("Rationale", value=struct.hypothesis.rationale, height=60)

                st.markdown("##### Methodology")
                approach = st.text_area("Approach", value=struct.methodology.approach, height=60)
                techniques = st.text_area(
                    "Techniques (one per line)",
                    value="\n".join(struct.methodology.techniques),
                    height=60,
                )

                st.markdown("##### Constraints")
                assumptions = st.text_area(
                    "Assumptions (one per line)",
                    value="\n".join(struct.constraints.assumptions),
                    height=60,
                )
                limitations = st.text_area(
                    "Limitations (one per line)",
                    value="\n".join(struct.constraints.limitations),
                    height=60,
                )

                st.markdown("##### Abstract Structure")
                variables = st.text_area(
                    "Variables (one per line)",
                    value="\n".join(struct.abstract_structure.variables),
                    height=60,
                )
                edges_json = st.text_area(
                    "Edges (JSON list)",
                    value=json.dumps(
                        [e.model_dump() for e in struct.abstract_structure.edges],
                        indent=2,
                    ),
                    height=100,
                )

                st.markdown("##### Reviewer Notes")
                notes = st.text_area("Notes", value=struct.reviewer_notes, height=60)

                save = st.form_submit_button("Save edits")

            if save:
                try:
                    edges_parsed = [CausalEdge(**e) for e in json.loads(edges_json)]
                except Exception:
                    edges_parsed = struct.abstract_structure.edges
                    st.warning("Could not parse edges JSON — kept previous value.")

                struct.problem = ProblemStatement(background=bg, problem=prob)
                struct.hypothesis = Hypothesis(statement=hyp_stmt, rationale=hyp_rat)
                struct.methodology = Methodology(
                    approach=approach,
                    techniques=[t.strip() for t in techniques.splitlines() if t.strip()],
                )
                struct.constraints = Constraints(
                    assumptions=[a.strip() for a in assumptions.splitlines() if a.strip()],
                    limitations=[l.strip() for l in limitations.splitlines() if l.strip()],
                )
                struct.abstract_structure = AbstractStructure(
                    variables=[v.strip() for v in variables.splitlines() if v.strip()],
                    edges=edges_parsed,
                )
                struct.reviewer_notes = notes
                st.session_state.structures[selected_id] = struct
                st.success("Edits saved.")

            # Action buttons
            btn_cols = st.columns(3)
            with btn_cols[0]:
                if st.button("✅ Approve", use_container_width=True):
                    struct.review_status = ReviewStatus.APPROVED
                    st.session_state.structures[selected_id] = struct
                    st.success("Paper **approved**.")
            with btn_cols[1]:
                if st.button("❌ Reject", use_container_width=True):
                    struct.review_status = ReviewStatus.REJECTED
                    st.session_state.structures[selected_id] = struct
                    st.error("Paper **rejected**.")
            with btn_cols[2]:
                if st.button("🔄 Re-Extract", use_container_width=True):
                    struct.review_status = ReviewStatus.PENDING
                    st.session_state.structures[selected_id] = struct
                    st.info("Marked for **re-extraction**.")

            st.caption(f"Current status: **{struct.review_status.value}**")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_structure(meta: PaperMeta) -> PaperStructure:
    """Generate a mock PaperStructure from paper metadata for demo purposes."""
    return PaperStructure(
        paper_id=meta.arxiv_id,
        title=meta.title,
        problem=ProblemStatement(
            background="(Auto-extracted from abstract) " + meta.summary[:200],
            problem="To be extracted by LLM.",
        ),
        hypothesis=Hypothesis(statement="To be extracted by LLM.", rationale=""),
        methodology=Methodology(approach="To be extracted by LLM.", techniques=[]),
        constraints=Constraints(assumptions=[], limitations=[]),
        abstract_structure=AbstractStructure(
            variables=["Variable_A", "Variable_B"],
            edges=[CausalEdge(source="Variable_A", target="Variable_B", relation="causes")],
        ),
    )
