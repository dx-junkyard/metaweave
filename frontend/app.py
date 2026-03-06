"""MetaWeave v1 — Streamlit frontend.

All data access goes through the FastAPI backend (BACKEND_URL env var).

非同期抽出の UX フロー:
1. "Fetch & Store" クリック → PDF ダウンロード後、抽出ジョブをキュー登録（即時返答）
2. 処理中の論文は processing_papers セッション状態で管理
3. st_autorefresh で 3 秒ごとにポーリング
4. 処理完了を検知したら st.toast で通知し、構造を自動ロード
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="MetaWeave v1", layout="wide", initial_sidebar_state="collapsed")

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
# 処理中の論文: arxiv_id -> {"title": str, "object_name": str}
if "processing_papers" not in st.session_state:
    st.session_state.processing_papers: dict[str, dict] = {}
# チャット履歴: arxiv_id -> list of {"role": str, "content": str}
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories: dict[str, list[dict]] = {}
# Auth
if "token" not in st.session_state:
    st.session_state.token: str | None = None
if "user_id" not in st.session_state:
    st.session_state.user_id: str | None = None
if "username" not in st.session_state:
    st.session_state.username: str | None = None
# 提案中の論文: arxiv_id -> proposal_id (直近に送信した提案を追跡)
if "pending_proposals" not in st.session_state:
    st.session_state.pending_proposals: dict[str, str] = {}
# ドラフトが存在する論文を追跡: arxiv_id -> True
if "draft_papers" not in st.session_state:
    st.session_state.draft_papers: dict[str, bool] = {}
# パターンプレビュー結果: arxiv_id -> pattern dict
if "pattern_preview" not in st.session_state:
    st.session_state.pattern_preview: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Auto-refresh while jobs are in flight
# ---------------------------------------------------------------------------

if st.session_state.processing_papers:
    st_autorefresh(interval=3_000, key="processing_autorefresh")


# ---------------------------------------------------------------------------
# Polling: check status of in-flight jobs and fire toasts
# ---------------------------------------------------------------------------

def _poll_processing_papers() -> None:
    """処理中の論文のステータスを確認し、完了 / 失敗を検知する。"""
    for arxiv_id, info in list(st.session_state.processing_papers.items()):
        try:
            status_data = api_extract_status(arxiv_id)
        except Exception:
            continue  # ネットワークエラーなどは無視して次のポーリングを待つ

        status = status_data.get("status", "")

        if status == "completed":
            # 構造を自動ロード: まずドラフトを試行、なければ正典
            _loaded_from_draft = False
            try:
                _draft = api_get_draft(arxiv_id)
                if _draft is not None:
                    st.session_state.structures[arxiv_id] = _draft["structure"]
                    st.session_state.draft_papers[arxiv_id] = True
                    _loaded_from_draft = True
            except Exception:
                pass
            if not _loaded_from_draft:
                try:
                    result = api_get_extract_result(arxiv_id)
                    st.session_state.structures[arxiv_id] = result
                    st.session_state.draft_papers.pop(arxiv_id, None)
                except Exception:
                    pass
            title = info.get("title", arxiv_id)
            del st.session_state.processing_papers[arxiv_id]
            st.toast(f"解析が完了しました！", icon="✅")
            st.rerun()

        elif status == "failed":
            error_msg = status_data.get("error", "不明なエラー")
            title = info.get("title", arxiv_id)
            del st.session_state.processing_papers[arxiv_id]
            st.error(f"「{title}」 の解析に失敗しました: {error_msg}")


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _auth_headers() -> dict:
    """Return Authorization header dict for the current session token."""
    return {"Authorization": f"Bearer {st.session_state.token}"}


def api_search(query: str, max_results: int) -> list[dict]:
    resp = requests.get(
        f"{BACKEND_URL}/api/search",
        params={"query": query, "max_results": max_results},
        headers=_auth_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def api_fetch(paper: dict) -> str:
    """POST /api/fetch and return the MinIO object name."""
    resp = requests.post(
        f"{BACKEND_URL}/api/fetch",
        json=paper,
        headers=_auth_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["object_name"]


def api_presigned_url(object_name: str) -> str:
    resp = requests.get(
        f"{BACKEND_URL}/api/presigned-url",
        params={"object_name": object_name},
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()["url"]


def api_list_papers() -> list[str]:
    resp = requests.get(
        f"{BACKEND_URL}/api/papers",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def api_extract(
    object_name: str, arxiv_id: str, is_draft: bool = False, user_id: str | None = None
) -> dict:
    """POST /api/extract — submit async job, returns {arxiv_id, status: "pending"}."""
    payload: dict = {"object_name": object_name, "arxiv_id": arxiv_id}
    if is_draft:
        payload["is_draft"] = True
        payload["user_id"] = user_id
    resp = requests.post(
        f"{BACKEND_URL}/api/extract",
        json=payload,
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def api_extract_status(arxiv_id: str) -> dict:
    """GET /api/extract-status/{arxiv_id} — poll job status."""
    resp = requests.get(
        f"{BACKEND_URL}/api/extract-status/{arxiv_id}",
        headers=_auth_headers(),
        timeout=10,
    )
    if resp.status_code == 404:
        return {"arxiv_id": arxiv_id, "status": "unknown"}
    resp.raise_for_status()
    return resp.json()


def api_get_extract_result(arxiv_id: str) -> dict:
    """GET /api/extract-result/{arxiv_id} — load a saved structure from MinIO."""
    resp = requests.get(
        f"{BACKEND_URL}/api/extract-result/{arxiv_id}",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def api_update_extract_result(arxiv_id: str, structure: dict) -> None:
    """PUT /api/extract-result/{arxiv_id} — persist an edited structure to MinIO."""
    resp = requests.put(
        f"{BACKEND_URL}/api/extract-result/{arxiv_id}",
        json=structure,
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()


def api_chat(arxiv_id: str, message: str, history: list[dict]) -> str:
    """POST /api/chat — RAG-based chat response (requires auth)."""
    resp = requests.post(
        f"{BACKEND_URL}/api/chat",
        json={"arxiv_id": arxiv_id, "message": message, "history": history},
        headers=_auth_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["answer"]


def api_get_chat_history(arxiv_id: str) -> list[dict]:
    """GET /api/chat/history/{arxiv_id} — load chat history from Neo4j."""
    resp = requests.get(
        f"{BACKEND_URL}/api/chat/history/{arxiv_id}",
        headers=_auth_headers(),
        timeout=10,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("history", [])


def api_propose_structure(arxiv_id: str, structure: dict) -> dict:
    """POST /api/propose-structure — submit edited structure for LLM gateway review."""
    resp = requests.post(
        f"{BACKEND_URL}/api/propose-structure",
        json={
            "arxiv_id": arxiv_id,
            "user_id": st.session_state.user_id,
            "proposed_structure": structure,
        },
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()  # {proposal_id, status: "pending"}


def api_get_proposals(arxiv_id: str) -> dict:
    """GET /api/proposals/{arxiv_id} — fetch proposal history from Neo4j."""
    resp = requests.get(
        f"{BACKEND_URL}/api/proposals/{arxiv_id}",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()  # {proposals: [...]}


def api_extract_pattern(arxiv_id: str) -> dict:
    """POST /api/patterns/extract/{arxiv_id} — preview pattern extraction (synchronous)."""
    resp = requests.post(
        f"{BACKEND_URL}/api/patterns/extract/{arxiv_id}",
        headers=_auth_headers(),
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def api_list_patterns() -> list[dict]:
    """GET /api/patterns — fetch all registered patterns."""
    resp = requests.get(
        f"{BACKEND_URL}/api/patterns",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("patterns", [])


def api_get_paper_patterns(arxiv_id: str) -> list[dict]:
    """GET /api/papers/{arxiv_id}/patterns — fetch patterns matching a paper."""
    resp = requests.get(
        f"{BACKEND_URL}/api/papers/{arxiv_id}/patterns",
        headers=_auth_headers(),
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json().get("matches", [])


def api_get_draft(arxiv_id: str) -> dict | None:
    """GET /api/draft/{arxiv_id} — fetch user's private draft from Neo4j.

    Returns the draft structure dict, or None if no draft exists (404).
    """
    resp = requests.get(
        f"{BACKEND_URL}/api/draft/{arxiv_id}",
        headers=_auth_headers(),
        timeout=15,
    )
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()  # {arxiv_id, structure: {...}}


def api_save_draft(arxiv_id: str, structure: dict) -> dict:
    """PUT /api/draft/{arxiv_id} — save structure as user's private draft."""
    resp = requests.put(
        f"{BACKEND_URL}/api/draft/{arxiv_id}",
        json={"structure": structure},
        headers=_auth_headers(),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def api_register_pattern(pattern: dict) -> dict:
    """POST /api/patterns/register — register a confirmed pattern to Qdrant/Neo4j."""
    resp = requests.post(
        f"{BACKEND_URL}/api/patterns/register",
        json={"pattern": pattern},
        headers=_auth_headers(),
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()  # {pattern_id, status: "registered"}


def _auth_post(path: str, payload: dict) -> dict:
    """Helper: POST to an auth endpoint and return the response JSON dict."""
    resp = requests.post(f"{BACKEND_URL}{path}", json=payload, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _fetch_me(token: str) -> dict:
    """GET /api/auth/me with a Bearer token."""
    resp = requests.get(
        f"{BACKEND_URL}/api/auth/me",
        headers={"Authorization": f"Bearer {token}"},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def _apply_auth(token: str) -> None:
    """Populate session state from a freshly obtained JWT."""
    me = _fetch_me(token)
    st.session_state.token = token
    st.session_state.user_id = me["id"]
    st.session_state.username = me["username"]


# ---------------------------------------------------------------------------
# Login / Register page
# ---------------------------------------------------------------------------

def _show_auth_page() -> None:
    """Render the full-screen login / register page and stop execution."""
    _, center, _ = st.columns([1, 2, 1])
    with center:
        st.markdown("# MetaWeave v1")
        st.markdown("---")
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            with st.form("login_form"):
                lg_user = st.text_input("Username")
                lg_pass = st.text_input("Password", type="password")
                lg_submit = st.form_submit_button(
                    "Login", use_container_width=True, type="primary"
                )
            if lg_submit:
                if lg_user and lg_pass:
                    try:
                        data = _auth_post("/api/auth/login", {"username": lg_user, "password": lg_pass})
                        _apply_auth(data["access_token"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Login failed: {exc}")
                else:
                    st.warning("Please enter username and password.")

        with tab_register:
            with st.form("register_form"):
                rg_user = st.text_input("Username", key="rg_user")
                rg_email = st.text_input("Email", key="rg_email")
                rg_pass = st.text_input("Password", type="password", key="rg_pass")
                rg_submit = st.form_submit_button(
                    "Register", use_container_width=True, type="primary"
                )
            if rg_submit:
                if rg_user and rg_email and rg_pass:
                    try:
                        data = _auth_post(
                            "/api/auth/register",
                            {"username": rg_user, "email": rg_email, "password": rg_pass},
                        )
                        _apply_auth(data["access_token"])
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Registration failed: {exc}")
                else:
                    st.warning("Please fill in all fields.")


# ---------------------------------------------------------------------------
# Polling: check status of in-flight jobs and fire toasts
# (called here, after all API helpers are defined)
# ---------------------------------------------------------------------------

_poll_processing_papers()


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
            "smiles_dsl": "",
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
# Auth gate — show login page and stop if not authenticated
# ---------------------------------------------------------------------------

if not st.session_state.token:
    _show_auth_page()
    st.stop()

# Top-right user info + logout button
_, _user_col = st.columns([6, 2])
with _user_col:
    st.caption(f"👤 **{st.session_state.username}**")
    if st.button("Logout", key="btn_logout", use_container_width=True):
        st.session_state.token = None
        st.session_state.user_id = None
        st.session_state.username = None
        st.rerun()

# ---------------------------------------------------------------------------
# Sidebar — Navigation + Paper list (Validation View)
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## MetaWeave v1")
    st.caption(f"👤 {st.session_state.username}")
    page = st.radio("Navigation", ["Harvester Dashboard", "Validation View", "Pattern Library"])

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

            # 処理中の論文には ⏳ アイコンを表示
            _is_processing = _aid in st.session_state.processing_papers
            if _is_processing:
                _icon = "⏳"
            else:
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
            arxiv_id = meta["arxiv_id"]
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
                    stored = st.session_state.stored_papers.get(arxiv_id)
                    is_processing = arxiv_id in st.session_state.processing_papers
                    if is_processing:
                        st.info("⏳ 解析中…")
                    elif stored:
                        st.success(f"Stored: {stored}")
                with cols[2]:
                    if st.button("Fetch & Store", key=f"fetch_{idx}"):
                        try:
                            with st.spinner("Downloading & storing PDF …"):
                                object_name = api_fetch(meta)
                                st.session_state.stored_papers[arxiv_id] = object_name
                                st.session_state.paper_metadata[arxiv_id] = meta
                            # 非同期抽出ジョブをキュー登録（即時返答）
                            api_extract(object_name, arxiv_id)
                            st.session_state.processing_papers[arxiv_id] = {
                                "title": meta.get("title", arxiv_id),
                                "object_name": object_name,
                            }
                            st.toast(f"⏳ 「{meta.get('title', arxiv_id)}」 の解析を開始しました")
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
            is_processing = active_id in st.session_state.processing_papers

            # ── ドラフト優先ロジック: まずユーザーのドラフトを取得、なければ正典を取得
            _is_draft_view = False
            if active_id not in st.session_state.structures:
                # ドラフトを試行
                try:
                    _draft_resp = api_get_draft(active_id)
                    if _draft_resp is not None:
                        st.session_state.structures[active_id] = _draft_resp["structure"]
                        st.session_state.draft_papers[active_id] = True
                except Exception:
                    pass
                # ドラフトがなければ MinIO の正典を取得
                if active_id not in st.session_state.structures:
                    try:
                        _loaded = api_get_extract_result(active_id)
                        st.session_state.structures[active_id] = _loaded
                        st.session_state.draft_papers.pop(active_id, None)
                    except Exception:
                        st.session_state.structures[active_id] = _empty_structure(active_id)

            s = st.session_state.structures[active_id]
            _is_draft_view = st.session_state.draft_papers.get(active_id, False)

            status = s.get("review_status", "pending")
            title = (
                st.session_state.paper_metadata.get(active_id, {}).get("title")
                or s.get("title")
                or active_id
            )

            # ── Header: prominent title + arXiv ID ───────────────────────────
            if is_processing:
                st.title(f"⏳ {title}")
                st.info("バックグラウンドで解析処理中です。完了次第自動的に更新されます。")
            else:
                st.title(title)

            # ── ドラフト / 正典バッジ ────────────────────────────────────────
            _badge_col, _status_col = st.columns([1, 3])
            with _badge_col:
                if _is_draft_view:
                    st.info("📝 あなたの未公開ドラフト")
                else:
                    st.success("🏛️ システム正規テンプレート")
            with _status_col:
                st.caption(
                    f"arXiv ID: `{active_id}`  |  Status: {_STATUS_LABEL.get(status, '🟡 Pending')}"
                )

            # ── Action buttons (right-aligned) ───────────────────────────────
            _, act1, act2, act3, act4 = st.columns([5, 1, 1, 1, 1])
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
                        api_extract(
                            object_name,
                            active_id,
                            is_draft=True,
                            user_id=st.session_state.user_id,
                        )
                        st.session_state.processing_papers[active_id] = {
                            "title": title,
                            "object_name": object_name,
                        }
                        st.toast(f"⏳ 「{title}」 の再解析を開始しました（ドラフトとして保存されます）")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Re-extraction failed: {exc}")
            with act4:
                if st.button("✨ Pattern", use_container_width=True, key="btn_pattern"):
                    try:
                        with st.spinner("パターンを抽出中…"):
                            _preview = api_extract_pattern(active_id)
                            st.session_state.pattern_preview[active_id] = _preview
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Pattern extraction failed: {exc}")
            # 💾 Save edits is inside the structure editor form below

            st.divider()

            # ── Pattern tags (matched patterns for this paper) ──────────────
            try:
                _paper_patterns = api_get_paper_patterns(active_id)
                if _paper_patterns:
                    _tag_cols = st.columns(min(len(_paper_patterns), 6))
                    for _pi, _pm in enumerate(_paper_patterns):
                        with _tag_cols[_pi % len(_tag_cols)]:
                            _conf = _pm.get("confidence_score", 0)
                            _pname = _pm.get("pattern_name") or _pm.get("pattern_id", "")[:8]
                            st.markdown(
                                f"<span style='background:#e0f7fa;padding:2px 8px;"
                                f"border-radius:12px;font-size:0.85em;'>"
                                f"🔗 {_pname} ({_conf:.0%})</span>",
                                unsafe_allow_html=True,
                            )
            except Exception:
                pass  # パターンがない場合はスキップ

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

            # ── Structure Editor + Chat ───────────────────────────────────────
            with edit_col:
                struct_tab, chat_tab, proposals_tab = st.tabs(
                    ["📄 Extracted Structure", "💬 Chat", "📋 提案履歴"]
                )

                # ── Structure Editor tab ──────────────────────────────────────
                with struct_tab:
                    if is_processing:
                        st.markdown(
                            "🔄 **解析処理中です。** 完了後に構造データが表示されます。"
                        )

                    with st.form(f"structure_form_{active_id}"):
                        tab_dsl, tab1, tab2, tab3 = st.tabs(
                            ["🧬 SMILES DSL", "Problem / Hypothesis", "Method / Constraints", "Raw Variables & Edges"]
                        )

                        with tab_dsl:
                            st.markdown("#### Abstract Structure — SMILES DSL")
                            smiles_dsl = st.text_area(
                                "MetaWeave-SMILES DSL",
                                value=s["abstract_structure"].get("smiles_dsl", ""),
                                height=200,
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
                            st.markdown("#### Raw Variables & Edges")
                            variables = st.text_area(
                                "Variables (one per line)",
                                value="\n".join(s["abstract_structure"]["variables"]),
                                height=110,
                            )
                            edges_json = st.text_area(
                                "Edges (JSON list)",
                                value=json.dumps(s["abstract_structure"]["edges"], indent=2, ensure_ascii=False),
                                height=150,
                            )
                            st.markdown("#### Reviewer Notes")
                            notes = st.text_area(
                                "Notes",
                                value=s.get("reviewer_notes", ""),
                                height=110,
                            )

                        _btn_col1, _btn_col2 = st.columns(2)
                        with _btn_col1:
                            save_draft_btn = st.form_submit_button(
                                "💾 Save edits",
                                use_container_width=True,
                            )
                        with _btn_col2:
                            propose = st.form_submit_button(
                                "💡 変更を提案する (Propose)",
                                use_container_width=True,
                                type="primary",
                            )

                    # ── フォーム送信後の共通処理: 編集内容を dict に変換 ────
                    if save_draft_btn or propose:
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
                                "smiles_dsl": smiles_dsl,
                                "variables": [
                                    v.strip() for v in variables.splitlines() if v.strip()
                                ],
                                "edges": edges_parsed,
                            },
                            "review_status": s.get("review_status", "pending"),
                            "reviewer_notes": notes,
                        }
                        st.session_state.structures[active_id] = updated

                    if save_draft_btn:
                        # ドラフトとして Neo4j に保存
                        try:
                            api_save_draft(active_id, updated)
                            st.session_state.draft_papers[active_id] = True
                            st.toast("ドラフトを保存しました", icon="💾")
                        except Exception as exc:
                            st.error(f"ドラフトの保存に失敗しました: {exc}")

                    if propose:
                        # LLM Gateway へ提案を送信
                        try:
                            result = api_propose_structure(active_id, updated)
                            proposal_id = result.get("proposal_id", "")
                            st.session_state.pending_proposals[active_id] = proposal_id
                            st.toast(
                                "提案を送信しました。AIがレビュー・マージを行います",
                                icon="💡",
                            )
                        except Exception as exc:
                            st.error(f"提案の送信に失敗しました: {exc}")

                    # ── パターンプレビュー表示 ──────────────────────────────
                    _preview_data = st.session_state.pattern_preview.get(active_id)
                    if _preview_data:
                        st.divider()
                        st.markdown("### ✨ パターン抽出プレビュー")
                        st.caption(
                            "抽出された抽象化パターンを確認・編集できます。"
                            "内容に納得したら「🌍 正式登録」ボタンで共有知として登録してください。"
                        )
                        with st.expander("パターンプレビュー", expanded=True):
                            _pv_name = st.text_input(
                                "パターン名",
                                value=_preview_data.get("name", ""),
                                key="pv_name",
                            )
                            _pv_desc = st.text_area(
                                "説明",
                                value=_preview_data.get("description", ""),
                                height=100,
                                key="pv_desc",
                            )
                            _pv_vars_raw = _preview_data.get("variables_template", [])
                            _pv_vars = st.text_area(
                                "変数テンプレート (1行1変数)",
                                value="\n".join(_pv_vars_raw),
                                height=80,
                                key="pv_vars",
                            )
                            _pv_rules_raw = _preview_data.get("structural_rules", [])
                            _pv_rules = st.text_area(
                                "構造ルール (1行1ルール)",
                                value="\n".join(_pv_rules_raw),
                                height=100,
                                key="pv_rules",
                            )

                            _reg_col, _cancel_col = st.columns(2)
                            with _reg_col:
                                if st.button(
                                    "🌍 このパターンをシステムに正式登録する",
                                    use_container_width=True,
                                    type="primary",
                                    key="btn_register_pattern",
                                ):
                                    _edited_pattern = {
                                        "pattern_id": _preview_data.get("pattern_id", ""),
                                        "name": _pv_name,
                                        "description": _pv_desc,
                                        "variables_template": [
                                            v.strip()
                                            for v in _pv_vars.splitlines()
                                            if v.strip()
                                        ],
                                        "structural_rules": [
                                            r.strip()
                                            for r in _pv_rules.splitlines()
                                            if r.strip()
                                        ],
                                        "source_arxiv_id": _preview_data.get(
                                            "source_arxiv_id", active_id
                                        ),
                                    }
                                    try:
                                        _reg_resp = api_register_pattern(_edited_pattern)
                                        st.session_state.pattern_preview.pop(active_id, None)
                                        st.toast(
                                            f"パターンを正式登録しました (ID: {_reg_resp.get('pattern_id', '')[:8]}…)",
                                            icon="🌍",
                                        )
                                        st.rerun()
                                    except Exception as exc:
                                        st.error(f"パターンの登録に失敗しました: {exc}")
                            with _cancel_col:
                                if st.button(
                                    "❌ キャンセル",
                                    use_container_width=True,
                                    key="btn_cancel_pattern",
                                ):
                                    st.session_state.pattern_preview.pop(active_id, None)
                                    st.rerun()

                # ── Chat tab ──────────────────────────────────────────────────
                with chat_tab:
                    st.markdown(
                        "論文の内容や解析結果についてAIに質問できます。"
                        " 回答はQdrantのベクトル検索と抽出済み構造データを参照して生成されます。"
                    )

                    # 論文ごとのチャット履歴を初期化（Neo4jから取得）
                    if active_id not in st.session_state.chat_histories:
                        try:
                            persisted = api_get_chat_history(active_id)
                            st.session_state.chat_histories[active_id] = persisted
                        except Exception:
                            st.session_state.chat_histories[active_id] = []

                    chat_history = st.session_state.chat_histories[active_id]

                    # 履歴をクリアするボタン
                    if chat_history:
                        if st.button("🗑️ Clear chat history", key=f"clear_chat_{active_id}"):
                            st.session_state.chat_histories[active_id] = []
                            st.rerun()

                    # 過去のメッセージを表示
                    for turn in chat_history:
                        with st.chat_message(turn["role"]):
                            st.markdown(turn["content"])

                    # 入力欄
                    user_input = st.chat_input(
                        "論文について質問してください…",
                        key=f"chat_input_{active_id}",
                        disabled=is_processing,
                    )

                    if is_processing:
                        st.info("解析処理完了後にチャットを利用できます。")
                    elif user_input:
                        # ユーザーメッセージを即座に表示
                        with st.chat_message("user"):
                            st.markdown(user_input)

                        # バックエンドを呼び出して回答を生成
                        with st.chat_message("assistant"):
                            with st.spinner("回答を生成中…"):
                                try:
                                    answer = api_chat(
                                        arxiv_id=active_id,
                                        message=user_input,
                                        history=chat_history,
                                    )
                                    st.markdown(answer)
                                    # 履歴に追加
                                    st.session_state.chat_histories[active_id].append(
                                        {"role": "user", "content": user_input}
                                    )
                                    st.session_state.chat_histories[active_id].append(
                                        {"role": "assistant", "content": answer}
                                    )
                                except Exception as exc:
                                    error_msg = f"チャットエラー: {exc}"
                                    st.error(error_msg)

                # ── Proposals History tab ─────────────────────────────────────
                with proposals_tab:
                    st.markdown("### 提案履歴 (LLM Gateway)")
                    st.caption(
                        "「💡 変更を提案する」で送信した編集内容とAIの審査・マージ結果を確認できます。"
                    )

                    # 直近提案のペンディング状態インジケーター
                    if active_id in st.session_state.pending_proposals:
                        st.info(
                            "⏳ **AIによるレビューが進行中です。** "
                            "しばらく後にこのページを更新してください。",
                            icon="⏳",
                        )

                    col_refresh, _ = st.columns([1, 4])
                    with col_refresh:
                        if st.button("🔄 更新", key=f"refresh_proposals_{active_id}"):
                            st.rerun()

                    try:
                        proposals_data = api_get_proposals(active_id)
                        proposals_list = proposals_data.get("proposals", [])
                    except Exception as exc:
                        st.error(f"提案履歴の取得に失敗しました: {exc}")
                        proposals_list = []

                    if not proposals_list:
                        st.info("この論文に対する提案履歴はまだありません。")
                    else:
                        _PROPOSAL_STATUS_ICON = {
                            "approved": "✅",
                            "failed": "❌",
                            "pending": "⏳",
                        }
                        for _p in proposals_list:
                            _p_status = _p.get("status", "pending")
                            _p_icon = _PROPOSAL_STATUS_ICON.get(_p_status, "🟡")
                            _p_id_short = _p.get("proposal_id", "")[:8]
                            _p_username = _p.get("username") or _p.get("user_id", "")
                            _p_created = _p.get("created_at", "")

                            # ペンディング解消チェック
                            _tracked_id = st.session_state.pending_proposals.get(active_id)
                            if (
                                _tracked_id == _p.get("proposal_id")
                                and _p_status in ("approved", "failed")
                            ):
                                st.session_state.pending_proposals.pop(active_id, None)

                            with st.container(border=True):
                                _head_col, _status_col = st.columns([3, 1])
                                with _head_col:
                                    st.markdown(
                                        f"**提案 `{_p_id_short}…`**  |  👤 {_p_username}"
                                    )
                                    if _p_created:
                                        st.caption(f"提出日時: {_p_created}")
                                with _status_col:
                                    st.markdown(f"## {_p_icon}")
                                    st.caption(_p_status.upper())

                                _reasoning = _p.get("evaluation_reasoning")
                                if _reasoning:
                                    with st.expander("AIの評価理由を見る"):
                                        st.markdown(_reasoning)
                                elif _p_status == "pending":
                                    st.caption("AIによる評価待ちです。")
                                else:
                                    st.caption("評価理由が記録されていません。")

# =========================================================================
# Page C — Pattern Library
# =========================================================================
elif page == "Pattern Library":
    st.header("Pattern Library")
    st.caption(
        "登録済みの抽象化パターン一覧と、各パターンに適合する論文群を確認できます。"
    )

    col_refresh_pl, _ = st.columns([1, 5])
    with col_refresh_pl:
        if st.button("🔄 更新", key="refresh_pattern_library"):
            st.rerun()

    try:
        _all_patterns = api_list_patterns()
    except Exception as exc:
        st.error(f"パターン一覧の取得に失敗しました: {exc}")
        _all_patterns = []

    if not _all_patterns:
        st.info(
            "まだ抽象化パターンが登録されていません。\n\n"
            "Validation View で論文を選択し、「✨ Pattern」ボタンをクリックして\n"
            "抽象化パターンを抽出してください。"
        )
    else:
        for _pat in _all_patterns:
            _pat_id = _pat.get("pattern_id", "")
            _pat_name = _pat.get("name", "Unnamed Pattern")
            _pat_desc = _pat.get("description", "")
            _pat_vars = _pat.get("variables_template", [])
            _pat_rules = _pat.get("structural_rules", [])
            _pat_source = _pat.get("source_arxiv_id", "")

            with st.container(border=True):
                _pl_left, _pl_right = st.columns([3, 1])
                with _pl_left:
                    st.markdown(f"### {_pat_name}")
                    st.markdown(_pat_desc)

                    if _pat_vars:
                        _vars_str = ", ".join(f"`{v}`" for v in _pat_vars)
                        st.markdown(f"**Variables:** {_vars_str}")
                    if _pat_rules:
                        st.markdown("**Structural Rules:**")
                        for _rule in _pat_rules:
                            st.markdown(f"- {_rule}")

                with _pl_right:
                    st.caption(f"ID: `{_pat_id[:8]}…`")
                    if _pat_source:
                        st.caption(f"Source: `{_pat_source}`")

                # このパターンに適合する論文を表示
                with st.expander("適合する論文を表示"):
                    try:
                        # Neo4j から MATCHES_PATTERN リレーションを逆引き
                        _match_resp = requests.get(
                            f"{BACKEND_URL}/api/patterns",
                            headers=_auth_headers(),
                            timeout=15,
                        )
                        # パターンIDでフィルタリングされた論文を取得するために
                        # 各論文のパターンマッチを確認する（簡易実装）
                        st.caption(
                            "このパターンに適合する論文は、各論文の詳細画面で "
                            "パターンタグとして表示されます。"
                        )
                    except Exception:
                        st.caption("情報の取得に失敗しました。")
