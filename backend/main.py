"""MetaWeave v1 — FastAPI backend.

Endpoints
---------
GET  /api/search                        Search arXiv for papers.
POST /api/search/structure              FANNS hybrid search (DSL regex + vector similarity).
POST /api/fetch                         Download a paper PDF and store it in MinIO.
POST /api/extract                       Submit async background extraction job.
GET  /api/extract-status/{arxiv_id}     Poll extraction job status (pending/processing/completed/failed).
GET  /api/extract-result/{arxiv_id}     Fetch a previously extracted paper structure from MinIO.
PUT  /api/extract-result/{arxiv_id}     Update an extracted paper structure in MinIO.
GET  /api/presigned-url                 Return a browser-accessible pre-signed URL for a stored PDF.
GET  /api/papers                        List all stored paper object names.
GET  /api/draft/{arxiv_id}              Get user's private draft from Neo4j.
PUT  /api/draft/{arxiv_id}              Save/upsert user's private draft in Neo4j.
POST /api/propose-structure             Submit a user structure proposal (saved to Neo4j, async LLM review).
POST /api/patterns/extract/{arxiv_id}   Preview pattern extraction (no DB save).
POST /api/patterns/register             Register a confirmed pattern to Qdrant/Neo4j.
POST /api/auth/register                 Register a new user (Neo4j User node, returns JWT).
POST /api/auth/login                    Authenticate and return a JWT.
GET  /api/auth/me                       Return the current user's profile (requires Bearer token).
GET  /api/patterns/{pattern_id}/suggestions  Missing link suggestions for a pattern.
GET  /healthz                           Health check.
"""

from __future__ import annotations

import datetime
import io
import json
import logging
import os
import threading
import uuid
from functools import lru_cache

import jwt
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from metaweave import extractor as ext
from metaweave.batch import run_pattern_evaluation_task
from metaweave.chat import generate_chat_response
from metaweave.db import get_driver
from metaweave.embedder import embed_and_store_pattern, search_fanns_hybrid
from metaweave.harvester import PaperMeta, fetch_and_store, search_arxiv
from metaweave.llm import generate_missing_link_suggestions, get_client, get_settings
from metaweave.schema import (
    AbstractionPattern,
    FieldSuggestion,
    MissingLinkSuggestion,
    PaperStructure,
    PatternMatch,
    StructureProposal,
)
from metaweave.storage import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# JWT / password-hashing configuration
# ---------------------------------------------------------------------------

_JWT_SECRET: str = os.environ.get("JWT_SECRET", "metaweave-dev-secret-change-in-prod")
_JWT_ALGORITHM: str = "HS256"
_JWT_EXPIRE_HOURS: int = 24

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer()

app = FastAPI(title="MetaWeave API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Shared StorageManager (initialised once on first use)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _storage() -> StorageManager:
    return StorageManager()


# ---------------------------------------------------------------------------
# In-memory extraction job status store
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()
_job_status: dict[str, "ExtractJobStatus"] = {}


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class PaperMetaOut(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    categories: list[str]
    pdf_url: str
    published: str
    license: str
    commercial_flag: bool


class FetchRequest(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    categories: list[str]
    pdf_url: str
    published: str
    license: str = ""
    commercial_flag: bool = False


class FetchResponse(BaseModel):
    object_name: str


class PresignedUrlResponse(BaseModel):
    url: str


class ExtractRequest(BaseModel):
    object_name: str
    arxiv_id: str
    is_draft: bool = False
    user_id: str | None = None
    skip_embedding: bool = False


class ExtractAccepted(BaseModel):
    """Immediate response for an accepted async extraction job."""

    arxiv_id: str
    status: str = "pending"


class ExtractJobStatus(BaseModel):
    """Current status of an async extraction job."""

    arxiv_id: str
    status: str  # pending | processing | completed | failed
    error: str | None = None


class ChatRequest(BaseModel):
    """Request body for the RAG chat endpoint."""

    arxiv_id: str
    message: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    """Response from the RAG chat endpoint."""

    answer: str


class ChatHistoryResponse(BaseModel):
    """Response from the chat history endpoint."""

    history: list[dict]


class ProposeStructureRequest(BaseModel):
    """Request body for POST /api/propose-structure."""

    arxiv_id: str
    user_id: str
    proposed_structure: PaperStructure


class ProposeStructureResponse(BaseModel):
    """Immediate response for an accepted structure proposal."""

    proposal_id: str
    status: str = "pending"


class ProposalItem(BaseModel):
    """Single proposal entry returned by GET /api/proposals/{arxiv_id}."""

    proposal_id: str
    user_id: str
    username: str | None = None
    status: str
    evaluation_reasoning: str | None = None
    created_at: str | None = None


class ProposalListResponse(BaseModel):
    """List of proposals for a given paper."""

    proposals: list[ProposalItem]


class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOut(BaseModel):
    id: str
    username: str
    email: str


# ---------------------------------------------------------------------------
# Pattern request / response models
# ---------------------------------------------------------------------------

class PatternExtractAccepted(BaseModel):
    """Immediate response for an accepted pattern extraction job."""

    arxiv_id: str
    status: str = "pending"


class PatternOut(BaseModel):
    """Public representation of an AbstractionPattern."""

    pattern_id: str
    name: str
    description: str
    variables_template: list[str]
    structural_rules: list[str]
    source_arxiv_id: str


class PatternListResponse(BaseModel):
    """List of patterns."""

    patterns: list[PatternOut]


class PatternMatchOut(BaseModel):
    """A pattern match result for a paper."""

    match_id: str
    pattern_id: str
    pattern_name: str | None = None
    target_arxiv_id: str
    mapping_explanation: str
    confidence_score: float


class PaperPatternListResponse(BaseModel):
    """List of pattern matches for a specific paper."""

    matches: list[PatternMatchOut]


class DraftResponse(BaseModel):
    """Response for GET /api/draft/{arxiv_id}."""

    arxiv_id: str
    structure: PaperStructure


class DraftSaveRequest(BaseModel):
    """Request body for PUT /api/draft/{arxiv_id}."""

    structure: PaperStructure


class PatternRegisterRequest(BaseModel):
    """Request body for POST /api/patterns/register."""

    pattern: AbstractionPattern


class SuggestionOut(BaseModel):
    """A single field suggestion for Missing Link."""

    field: str
    reasoning: str
    keywords: list[str]


class MissingLinkSuggestionResponse(BaseModel):
    """Response for GET /api/patterns/{pattern_id}/suggestions."""

    pattern_id: str
    pattern_name: str
    suggestions: list[SuggestionOut]
    cached: bool = False


class PatternRegisterResponse(BaseModel):
    """Response for POST /api/patterns/register."""

    pattern_id: str
    status: str = "registered"


# ---------------------------------------------------------------------------
# FANNS structure search request / response models
# ---------------------------------------------------------------------------

class StructureSearchRequest(BaseModel):
    """Request body for POST /api/search/structure."""

    query_dsl_regex: str = ""
    query_text: str = ""
    top_k: int = 5


class StructureSearchHit(BaseModel):
    """A single result from the FANNS hybrid search."""

    arxiv_id: str
    score: float
    text: str
    smiles_dsl: str = ""
    variables: list[str] = []


class StructureSearchResponse(BaseModel):
    """Response for POST /api/search/structure."""

    hits: list[StructureSearchHit]
    total: int


# ---------------------------------------------------------------------------
# Auth utility functions
# ---------------------------------------------------------------------------

def _hash_password(plain: str) -> str:
    return _pwd_context.hash(plain)


def _verify_password(plain: str, hashed: str) -> bool:
    return _pwd_context.verify(plain, hashed)


def _create_token(user_id: str, username: str, email: str) -> str:
    expire = datetime.datetime.utcnow() + datetime.timedelta(hours=_JWT_EXPIRE_HOURS)
    payload = {"sub": user_id, "username": username, "email": email, "exp": expire}
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGORITHM)


def _get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """FastAPI dependency: decode Bearer token and return user dict."""
    try:
        payload = jwt.decode(
            credentials.credentials, _JWT_SECRET, algorithms=[_JWT_ALGORITHM]
        )
        return {
            "id": payload["sub"],
            "username": payload["username"],
            "email": payload["email"],
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ---------------------------------------------------------------------------
# Background extraction task
# ---------------------------------------------------------------------------

def _run_extraction_task(
    object_name: str,
    arxiv_id: str,
    is_draft: bool = False,
    user_id: str | None = None,
    skip_embedding: bool = False,
) -> None:
    """バックグラウンドで実行される抽出タスク。

    1. ステータスを "processing" に更新
    2. MinIO から PDF を取得
    3. テキスト抽出 → 仮説検証型チャンク解析 → Embedding（並行、skip_embedding=False の場合）
    4. is_draft=False の場合は MinIO に保存（従来どおり）、
       is_draft=True の場合は Neo4j のユーザードラフトとして保存
    5. ステータスを "completed" または "failed" に更新
    """
    with _job_lock:
        _job_status[arxiv_id] = ExtractJobStatus(arxiv_id=arxiv_id, status="processing")

    try:
        # 1. Fetch PDF from MinIO
        logger.info(
            "Background extraction started for %s (is_draft=%s, skip_embedding=%s)",
            arxiv_id, is_draft, skip_embedding,
        )
        response = _storage().client.get_object("raw-papers", object_name)
        pdf_bytes = response.read()
        response.close()
        response.release_conn()

        # 2. Extract text and structure (Embedding は skip_embedding=False の場合のみ実行)
        #    GROBID ベースの論理チャンク生成を優先し、失敗時は text フォールバック
        text = ext.extract_text_from_pdf_bytes(pdf_bytes)
        structure = ext.extract_paper_structure(
            text, paper_id=arxiv_id, skip_embedding=skip_embedding, pdf_bytes=pdf_bytes,
        )

        if is_draft and user_id:
            # 3a. ドラフトモード: Neo4j のユーザードラフトとして保存
            structure_json = structure.model_dump_json()
            driver = get_driver()
            with driver.session() as session:
                session.run(
                    """
                    MERGE (u:User {id: $user_id})
                    MERGE (d:Draft {arxiv_id: $arxiv_id, user_id: $user_id})
                    MERGE (u)-[:HAS_DRAFT]->(d)
                    SET d.structure = $structure_json,
                        d.updated_at = $updated_at
                    """,
                    user_id=user_id,
                    arxiv_id=arxiv_id,
                    structure_json=structure_json,
                    updated_at=datetime.datetime.utcnow().isoformat(),
                )
            logger.info("Draft extraction saved to Neo4j for %s (user=%s)", arxiv_id, user_id)
        else:
            # 3b. 通常モード: MinIO の extracted-structures バケットに保存
            json_bytes = structure.model_dump_json().encode()
            safe_id = arxiv_id.replace("/", "_")
            _storage().client.put_object(
                "extracted-structures",
                f"{safe_id}.json",
                io.BytesIO(json_bytes),
                length=len(json_bytes),
                content_type="application/json",
            )
            logger.info("Background extraction completed for %s", arxiv_id)

        with _job_lock:
            _job_status[arxiv_id] = ExtractJobStatus(arxiv_id=arxiv_id, status="completed")

    except Exception as exc:
        logger.exception("Background extraction failed for %s", arxiv_id)
        with _job_lock:
            _job_status[arxiv_id] = ExtractJobStatus(
                arxiv_id=arxiv_id, status="failed", error=str(exc)
            )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/api/search", response_model=list[PaperMetaOut])
def search(
    query: str = Query(..., description="arXiv search query (e.g. cat:cs.AI)"),
    max_results: int = Query(default=20, ge=1, le=100),
) -> list[PaperMetaOut]:
    """Search arXiv and return paper metadata with commercial-publisher flags."""
    try:
        results = search_arxiv(query, max_results=max_results)
    except Exception as exc:
        logger.exception("arXiv search failed")
        raise HTTPException(status_code=502, detail=f"arXiv search failed: {exc}") from exc

    return [
        PaperMetaOut(
            arxiv_id=m.arxiv_id,
            title=m.title,
            authors=m.authors,
            summary=m.summary,
            categories=m.categories,
            pdf_url=m.pdf_url,
            published=m.published,
            license=m.license,
            commercial_flag=m.commercial_flag,
        )
        for m in results
    ]


@app.post("/api/search/structure", response_model=StructureSearchResponse)
def search_structure(body: StructureSearchRequest) -> StructureSearchResponse:
    """FANNS ハイブリッド検索エンドポイント。

    正規表現ベースの DSL パターンフィルタと自然言語クエリのベクトル検索を
    組み合わせて、構造的に類似する論文を返す。

    - ``query_dsl_regex``: SMILES DSL ペイロードに対する正規表現（例: ``"Agent.*Resource"``）
    - ``query_text``: 意味的類似度検索用の自然言語クエリ
    - ``top_k``: 返却する上位件数（デフォルト 5）

    少なくとも ``query_dsl_regex`` か ``query_text`` のいずれか一方は必須。
    """
    if not body.query_dsl_regex and not body.query_text:
        raise HTTPException(
            status_code=400,
            detail="At least one of 'query_dsl_regex' or 'query_text' must be provided.",
        )

    try:
        results = search_fanns_hybrid(
            query_dsl_regex=body.query_dsl_regex,
            query_text=body.query_text,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("FANNS structure search failed")
        raise HTTPException(
            status_code=500, detail=f"Structure search failed: {exc}"
        ) from exc

    hits = [
        StructureSearchHit(
            arxiv_id=r["arxiv_id"],
            score=r["score"],
            text=r["text"],
            smiles_dsl=r.get("smiles_dsl", ""),
            variables=r.get("variables", []),
        )
        for r in results
    ]

    return StructureSearchResponse(hits=hits, total=len(hits))


# ---------------------------------------------------------------------------
# Natural Language → SMILES DSL conversion
# ---------------------------------------------------------------------------

class NlToDslRequest(BaseModel):
    """Request body for POST /api/search/nl-to-dsl."""

    natural_language_query: str


class NlToDslResponse(BaseModel):
    """Response for POST /api/search/nl-to-dsl."""

    query_dsl_regex: str
    explanation: str


_NL_TO_DSL_PROMPT = """\
あなたは MetaWeave の構造検索アシスタントです。
ユーザーが自然言語で述べた課題・疑問を、MetaWeave-SMILES DSL の正規表現パターンに変換してください。

## MetaWeave-SMILES DSL の書式
- 変数: [略号:概念名:オントロジータイプ]  例: [a:Agent:Organization]
- 因果辺: -[relation:polarity]->  例: -[cause:+]->
- オントロジータイプ: Agent, Resource, Event, Purpose, InstitutionalAgent, IntentionalMoment

## 出力ルール
1. Qdrant の payload テキスト検索 (部分一致) に適した正規表現を生成する
2. ワイルドカード (.*) を活用し、具体的なドメイン用語ではなく抽象的な構造パターンを捉える
3. 複数のパターンが考えられる場合は、最も包括的な1つを選ぶ
4. 出力は JSON 形式で返す: {"query_dsl_regex": "<正規表現>", "explanation": "<日本語での簡潔な説明>"}

## 例
入力: "限られた資源を複数の主体が奪い合う問題"
出力: {"query_dsl_regex": "\\\\[.*:Agent\\\\].*-\\\\[.*compete.*\\\\]->.*\\\\[.*:Resource\\\\]", "explanation": "複数のエージェントがリソースを巡って競合する構造パターン"}

入力: "技術革新が既存のビジネスモデルを破壊する現象"
出力: {"query_dsl_regex": "\\\\[.*:Event\\\\].*-\\\\[.*destroy.*:-\\\\]->.*\\\\[.*:Resource\\\\]", "explanation": "イベント（技術革新）がリソース（ビジネスモデル）を負の方向に変化させる構造"}
"""


@app.post("/api/search/nl-to-dsl", response_model=NlToDslResponse)
def nl_to_dsl(body: NlToDslRequest) -> NlToDslResponse:
    """自然言語クエリを MetaWeave-SMILES DSL 正規表現に変換する。

    LLM を使い、ユーザーの課題記述を Qdrant ペイロード検索互換の
    DSL 正規表現に変換して返す。
    """
    if not body.natural_language_query.strip():
        raise HTTPException(status_code=400, detail="natural_language_query is required.")

    client = get_client()
    settings = get_settings()

    try:
        response = client.chat.completions.create(
            model=settings.analysis_model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{_NL_TO_DSL_PROMPT}\n\n"
                        f"入力: \"{body.natural_language_query}\"\n"
                        "出力:"
                    ),
                },
            ],
        )
        raw = response.choices[0].message.content.strip()

        # JSON 部分を抽出（コードブロック記法に対応）
        if "```" in raw:
            # ```json ... ``` or ``` ... ```
            import re as _re
            _match = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.DOTALL)
            if _match:
                raw = _match.group(1)
        elif raw.startswith("{"):
            pass  # already JSON
        else:
            # 行内の最初の { ... } を抜き出す
            import re as _re
            _match = _re.search(r"\{.*\}", raw, _re.DOTALL)
            if _match:
                raw = _match.group(0)

        parsed = json.loads(raw)
        return NlToDslResponse(
            query_dsl_regex=parsed.get("query_dsl_regex", ""),
            explanation=parsed.get("explanation", ""),
        )
    except json.JSONDecodeError:
        logger.warning("NL-to-DSL: LLM response was not valid JSON: %s", raw)
        # フォールバック: raw テキストをそのまま regex として返す
        return NlToDslResponse(
            query_dsl_regex=raw[:500] if raw else "",
            explanation="LLM出力のJSON解析に失敗しました。手動で調整してください。",
        )
    except Exception as exc:
        logger.exception("NL-to-DSL conversion failed")
        raise HTTPException(
            status_code=500, detail=f"NL-to-DSL conversion failed: {exc}"
        ) from exc


@app.post("/api/fetch", response_model=FetchResponse)
def fetch(body: FetchRequest) -> FetchResponse:
    """Download the PDF for the specified paper and store it in MinIO."""
    meta = PaperMeta(
        arxiv_id=body.arxiv_id,
        title=body.title,
        authors=body.authors,
        summary=body.summary,
        categories=body.categories,
        pdf_url=body.pdf_url,
        published=body.published,
        license=body.license,
        commercial_flag=body.commercial_flag,
    )
    try:
        object_name = fetch_and_store(meta, _storage())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("PDF fetch/store failed")
        raise HTTPException(status_code=502, detail=f"Fetch failed: {exc}") from exc

    return FetchResponse(object_name=object_name)


@app.post("/api/extract", response_model=ExtractAccepted)
def extract(body: ExtractRequest, background_tasks: BackgroundTasks) -> ExtractAccepted:
    """Submit an async background extraction job for a stored PDF.

    Returns immediately with status="pending".
    Poll ``GET /api/extract-status/{arxiv_id}`` to track progress.

    When ``is_draft=True`` and ``user_id`` is provided, the extraction result is
    saved as a private Neo4j draft instead of overwriting the canonical MinIO store.
    """
    with _job_lock:
        _job_status[body.arxiv_id] = ExtractJobStatus(
            arxiv_id=body.arxiv_id, status="pending"
        )
    # Re-Extract（is_draft=True）の場合は Qdrant チャンクが既存のためスキップ
    effective_skip_embedding = body.skip_embedding or body.is_draft
    background_tasks.add_task(
        _run_extraction_task,
        body.object_name,
        body.arxiv_id,
        is_draft=body.is_draft,
        user_id=body.user_id,
        skip_embedding=effective_skip_embedding,
    )
    logger.info(
        "Extraction job queued for %s (is_draft=%s, skip_embedding=%s)",
        body.arxiv_id, body.is_draft, effective_skip_embedding,
    )
    return ExtractAccepted(arxiv_id=body.arxiv_id, status="pending")


@app.get("/api/extract-status/{arxiv_id}", response_model=ExtractJobStatus)
def get_extract_status(arxiv_id: str) -> ExtractJobStatus:
    """Return the current status of a background extraction job.

    If no in-memory record exists but a result file is found in MinIO,
    returns status="completed" (supports restarts / cross-session queries).
    """
    with _job_lock:
        status = _job_status.get(arxiv_id)

    if status is not None:
        return status

    # Check MinIO for a persisted result (e.g. server restarted after completion)
    safe_id = arxiv_id.replace("/", "_")
    try:
        _storage().client.stat_object("extracted-structures", f"{safe_id}.json")
        return ExtractJobStatus(arxiv_id=arxiv_id, status="completed")
    except Exception:
        raise HTTPException(
            status_code=404, detail=f"No extraction job found for '{arxiv_id}'"
        )


@app.get("/api/presigned-url", response_model=PresignedUrlResponse)
def presigned_url(
    object_name: str = Query(..., description="MinIO object name, e.g. arxiv/2024/2401.12345.pdf"),
    bucket: str = Query(default="raw-papers"),
) -> PresignedUrlResponse:
    """Return a browser-accessible pre-signed URL for the given object."""
    try:
        url = _storage().presigned_url(bucket, object_name)
    except Exception as exc:
        logger.exception("Pre-signed URL generation failed")
        raise HTTPException(status_code=500, detail=f"URL generation failed: {exc}") from exc

    return PresignedUrlResponse(url=url)


@app.get("/api/papers", response_model=list[str])
def list_papers(
    prefix: str = Query(default="", description="Object prefix filter"),
) -> list[str]:
    """List stored paper object names in the raw-papers bucket."""
    try:
        return _storage().list_objects("raw-papers", prefix=prefix)
    except Exception as exc:
        logger.exception("Listing objects failed")
        raise HTTPException(status_code=500, detail=f"List failed: {exc}") from exc


@app.get("/api/extract-result/{arxiv_id}", response_model=PaperStructure)
def get_extract_result(arxiv_id: str) -> PaperStructure:
    """Fetch a previously extracted paper structure from the extracted-structures bucket."""
    safe_id = arxiv_id.replace("/", "_")
    try:
        response = _storage().client.get_object("extracted-structures", f"{safe_id}.json")
        data = response.read()
        response.close()
        response.release_conn()
    except Exception as exc:
        raise HTTPException(
            status_code=404, detail=f"No extraction found for '{arxiv_id}'"
        ) from exc
    try:
        return PaperStructure.model_validate_json(data)
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse stored structure: {exc}"
        ) from exc


@app.put("/api/extract-result/{arxiv_id}", response_model=PaperStructure)
def update_extract_result(arxiv_id: str, body: PaperStructure) -> PaperStructure:
    """Persist an updated paper structure to the extracted-structures bucket."""
    safe_id = arxiv_id.replace("/", "_")
    try:
        json_bytes = body.model_dump_json().encode()
        _storage().client.put_object(
            "extracted-structures",
            f"{safe_id}.json",
            io.BytesIO(json_bytes),
            length=len(json_bytes),
            content_type="application/json",
        )
    except Exception as exc:
        logger.exception("Failed to update extract result in MinIO")
        raise HTTPException(status_code=500, detail=f"Update failed: {exc}") from exc
    return body


@app.get("/api/draft/{arxiv_id}", response_model=DraftResponse)
def get_draft(
    arxiv_id: str,
    current_user: dict = Depends(_get_current_user),
) -> DraftResponse:
    """ログインユーザーの指定論文に対するプライベートドラフトを Neo4j から取得する。"""
    driver = get_driver()
    with driver.session() as session:
        record = session.run(
            """
            MATCH (u:User {id: $user_id})-[:HAS_DRAFT]->(d:Draft {arxiv_id: $arxiv_id})
            RETURN d.structure AS structure
            """,
            user_id=current_user["id"],
            arxiv_id=arxiv_id,
        ).single()

    if not record or not record["structure"]:
        raise HTTPException(status_code=404, detail="No draft found")

    try:
        structure = PaperStructure.model_validate_json(record["structure"])
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse draft structure: {exc}"
        ) from exc

    return DraftResponse(arxiv_id=arxiv_id, structure=structure)


@app.put("/api/draft/{arxiv_id}", response_model=DraftResponse)
def save_draft(
    arxiv_id: str,
    body: DraftSaveRequest,
    current_user: dict = Depends(_get_current_user),
) -> DraftResponse:
    """ログインユーザーのドラフトを Neo4j に保存（UPSERT）する。"""
    structure_json = body.structure.model_dump_json()
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (u:User {id: $user_id})
            MERGE (d:Draft {arxiv_id: $arxiv_id, user_id: $user_id})
            MERGE (u)-[:HAS_DRAFT]->(d)
            SET d.structure = $structure_json,
                d.updated_at = $updated_at
            """,
            user_id=current_user["id"],
            arxiv_id=arxiv_id,
            structure_json=structure_json,
            updated_at=datetime.datetime.utcnow().isoformat(),
        )
    logger.info("Draft saved for user=%s arxiv_id=%s", current_user["id"], arxiv_id)
    return DraftResponse(arxiv_id=arxiv_id, structure=body.structure)


@app.post("/api/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    current_user: dict = Depends(_get_current_user),
) -> ChatResponse:
    """RAG-based chat endpoint (requires authentication).

    ユーザーの質問を受け取り、Qdrant のベクトル検索と MinIO の PaperStructure を
    組み合わせて LLM に回答を生成させる。回答後、チャット履歴を Neo4j に永続化する。
    """
    try:
        answer = generate_chat_response(
            arxiv_id=body.arxiv_id,
            message=body.message,
            history=body.history,
            minio_client=_storage().client,
        )
    except Exception as exc:
        logger.exception("Chat generation failed for %s", body.arxiv_id)
        raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

    # Persist updated chat history to Neo4j
    updated_history = body.history + [
        {"role": "user", "content": body.message},
        {"role": "assistant", "content": answer},
    ]
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MERGE (p:Paper {arxiv_id: $arxiv_id})
                MERGE (u)-[r:CHATTED_ABOUT]->(p)
                SET r.history = $history
                """,
                user_id=current_user["id"],
                arxiv_id=body.arxiv_id,
                history=json.dumps(updated_history, ensure_ascii=False),
            )
    except Exception:
        logger.exception(
            "Failed to persist chat history for user=%s arxiv_id=%s",
            current_user["id"],
            body.arxiv_id,
        )

    return ChatResponse(answer=answer)


@app.get("/api/chat/history/{arxiv_id}", response_model=ChatHistoryResponse)
def get_chat_history(
    arxiv_id: str,
    current_user: dict = Depends(_get_current_user),
) -> ChatHistoryResponse:
    """ログイン中のユーザーの特定論文に対するチャット履歴を Neo4j から取得して返す。"""
    driver = get_driver()
    with driver.session() as session:
        record = session.run(
            """
            MATCH (u:User {id: $user_id})-[r:CHATTED_ABOUT]->(p:Paper {arxiv_id: $arxiv_id})
            RETURN r.history AS history
            """,
            user_id=current_user["id"],
            arxiv_id=arxiv_id,
        ).single()

    if not record or not record["history"]:
        return ChatHistoryResponse(history=[])

    try:
        history = json.loads(record["history"])
    except Exception:
        history = []

    return ChatHistoryResponse(history=history)


# ---------------------------------------------------------------------------
# Background LLM review task
# ---------------------------------------------------------------------------

def _run_review_task(proposal: StructureProposal) -> None:
    """バックグラウンドで実行される LLM 提案レビュータスク。

    1. MinIO から正典構造を取得（なければ提案構造を正典として扱う）
    2. evaluate_and_merge_proposals でマージ評価
    3. Neo4j の提案ノードに結果を書き戻す
    """
    driver = get_driver()

    # 1. 正典構造を MinIO から取得
    safe_id = proposal.arxiv_id.replace("/", "_")
    try:
        response = _storage().client.get_object("extracted-structures", f"{safe_id}.json")
        data = response.read()
        response.close()
        response.release_conn()
        base_structure = PaperStructure.model_validate_json(data)
    except Exception:
        logger.warning(
            "Base structure not found for %s — using proposal as baseline",
            proposal.arxiv_id,
        )
        base_structure = proposal.proposed_structure

    # 2. LLM によるマージ評価
    try:
        merge_result = ext.evaluate_and_merge_proposals(base_structure, proposal.proposed_structure)
        new_status = "approved"
        reasoning = merge_result.evaluation_reasoning
        merged_json = merge_result.merged_structure.model_dump_json()
    except Exception:
        logger.exception("LLM review failed for proposal %s", proposal.proposal_id)
        new_status = "failed"
        reasoning = "LLM review encountered an error."
        merged_json = ""

    # 3. 承認された場合、MinIO の正典構造を merged_structure で上書き更新
    if new_status == "approved" and merged_json:
        try:
            merged_bytes = merged_json.encode()
            _storage().client.put_object(
                "extracted-structures",
                f"{safe_id}.json",
                io.BytesIO(merged_bytes),
                length=len(merged_bytes),
                content_type="application/json",
            )
            logger.info("Updated canonical structure in MinIO for %s", proposal.arxiv_id)
        except Exception:
            logger.exception(
                "Failed to update MinIO with merged structure for %s", proposal.arxiv_id
            )

    # 4. Neo4j の提案ノードを更新
    with driver.session() as session:
        session.run(
            """
            MATCH (p:StructureProposal {proposal_id: $proposal_id})
            SET p.status = $status,
                p.evaluation_reasoning = $reasoning,
                p.merged_structure = $merged_json
            """,
            proposal_id=proposal.proposal_id,
            status=new_status,
            reasoning=reasoning,
            merged_json=merged_json,
        )
    logger.info(
        "Review task completed for proposal %s — status=%s",
        proposal.proposal_id,
        new_status,
    )


# ---------------------------------------------------------------------------
# Proposal endpoint
# ---------------------------------------------------------------------------

@app.post("/api/propose-structure", response_model=ProposeStructureResponse)
def propose_structure(
    body: ProposeStructureRequest, background_tasks: BackgroundTasks
) -> ProposeStructureResponse:
    """ユーザーの構造提案を受け付け、Neo4j に保存してバックグラウンドでレビューをトリガーする。

    処理フロー:
    1. 新しい proposal_id を生成
    2. Neo4j に User ノードと StructureProposal ノードを MERGE/CREATE
    3. BackgroundTasks に LLM レビュータスクを追加
    4. {proposal_id, status: "pending"} を即時返却
    """
    proposal_id = str(uuid.uuid4())
    proposal = StructureProposal(
        proposal_id=proposal_id,
        arxiv_id=body.arxiv_id,
        user_id=body.user_id,
        proposed_structure=body.proposed_structure,
    )

    # Neo4j への保存
    created_at = datetime.datetime.utcnow().isoformat()
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (u:User {id: $user_id})
                CREATE (p:StructureProposal {
                    proposal_id:        $proposal_id,
                    arxiv_id:           $arxiv_id,
                    user_id:            $user_id,
                    proposed_structure: $proposed_structure,
                    status:             'pending',
                    created_at:         $created_at
                })
                CREATE (u)-[:PROPOSED]->(p)
                """,
                user_id=body.user_id,
                proposal_id=proposal_id,
                arxiv_id=body.arxiv_id,
                proposed_structure=body.proposed_structure.model_dump_json(),
                created_at=created_at,
            )
    except Exception as exc:
        logger.exception("Failed to save proposal %s to Neo4j", proposal_id)
        raise HTTPException(status_code=500, detail=f"Failed to save proposal: {exc}") from exc

    # バックグラウンドで LLM レビューをトリガー
    background_tasks.add_task(_run_review_task, proposal)
    logger.info("Proposal %s queued for LLM review (arxiv_id=%s)", proposal_id, body.arxiv_id)

    return ProposeStructureResponse(proposal_id=proposal_id, status="pending")


@app.get("/api/proposals/{arxiv_id}", response_model=ProposalListResponse)
def get_proposals(
    arxiv_id: str,
    current_user: dict = Depends(_get_current_user),
) -> ProposalListResponse:
    """指定論文に対する全提案履歴を Neo4j から取得して返す。

    StructureProposal ノードを arxiv_id で絞り込み、提案者のユーザー名・ステータス・
    LLM 評価理由などをリスト形式で返す。
    """
    driver = get_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (u:User)-[:PROPOSED]->(p:StructureProposal {arxiv_id: $arxiv_id})
            RETURN p.proposal_id       AS proposal_id,
                   p.user_id           AS user_id,
                   u.username          AS username,
                   p.status            AS status,
                   p.evaluation_reasoning AS evaluation_reasoning,
                   p.created_at        AS created_at
            ORDER BY p.created_at DESC
            """,
            arxiv_id=arxiv_id,
        ).data()

    proposals = [
        ProposalItem(
            proposal_id=r["proposal_id"],
            user_id=r["user_id"],
            username=r.get("username"),
            status=r["status"] or "pending",
            evaluation_reasoning=r.get("evaluation_reasoning"),
            created_at=r.get("created_at"),
        )
        for r in records
    ]
    return ProposalListResponse(proposals=proposals)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/api/auth/register", response_model=TokenResponse, status_code=201)
def auth_register(body: RegisterRequest) -> TokenResponse:
    """新規ユーザーを Neo4j に登録し、JWT を返す。"""
    driver = get_driver()
    with driver.session() as session:
        existing = session.run(
            "MATCH (u:User {username: $username}) RETURN u.id AS id LIMIT 1",
            username=body.username,
        ).single()
        if existing:
            raise HTTPException(status_code=409, detail="Username already taken")

        user_id = str(uuid.uuid4())
        hashed_pw = _hash_password(body.password)
        session.run(
            """
            CREATE (u:User {
                id:              $id,
                username:        $username,
                email:           $email,
                hashed_password: $hashed_password
            })
            """,
            id=user_id,
            username=body.username,
            email=body.email,
            hashed_password=hashed_pw,
        )

    logger.info("Registered new user '%s' (id=%s)", body.username, user_id)
    return TokenResponse(access_token=_create_token(user_id, body.username, body.email))


@app.post("/api/auth/login", response_model=TokenResponse)
def auth_login(body: LoginRequest) -> TokenResponse:
    """ユーザー名とパスワードを検証し、JWT を返す。"""
    driver = get_driver()
    with driver.session() as session:
        record = session.run(
            "MATCH (u:User {username: $username}) "
            "RETURN u.id AS id, u.email AS email, u.hashed_password AS hashed_password",
            username=body.username,
        ).single()

    if not record or not _verify_password(body.password, record["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return TokenResponse(
        access_token=_create_token(record["id"], body.username, record["email"])
    )


@app.get("/api/auth/me", response_model=UserOut)
def auth_me(current_user: dict = Depends(_get_current_user)) -> UserOut:
    """Bearer トークンからデコードした現在のユーザー情報を返す。"""
    return UserOut(**current_user)


# ---------------------------------------------------------------------------
# Background pattern extraction + batch evaluation task
# ---------------------------------------------------------------------------

def _run_pattern_evaluation_task(pattern: AbstractionPattern) -> None:
    """バックグラウンドで実行されるパターン再評価バッチタスク。

    登録済みパターンに対して、過去の論文への構造的同型性を評価する。
    """
    try:
        matches = run_pattern_evaluation_task(pattern, _storage().client)
        logger.info(
            "Batch evaluation completed for pattern %s: %d matches",
            pattern.pattern_id, len(matches),
        )
    except Exception:
        logger.exception("Batch evaluation failed for pattern %s", pattern.pattern_id)


# ---------------------------------------------------------------------------
# Pattern API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/patterns/extract/{arxiv_id}", response_model=AbstractionPattern)
def extract_pattern_preview(
    arxiv_id: str,
) -> AbstractionPattern:
    """指定論文から抽象化パターンを抽出し、プレビュー用にレスポンスとして返す。

    DB（Qdrant / Neo4j）には保存しない。ユーザーが結果を確認・修正した後、
    ``POST /api/patterns/register`` で正式登録する。
    """
    safe_id = arxiv_id.replace("/", "_")
    try:
        response = _storage().client.get_object("extracted-structures", f"{safe_id}.json")
        data = response.read()
        response.close()
        response.release_conn()
        structure = PaperStructure.model_validate_json(data)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"No extracted structure found for '{arxiv_id}'. Extract the paper first.",
        )

    try:
        pattern = ext.extract_abstraction_pattern(structure)
    except Exception as exc:
        logger.exception("Pattern extraction preview failed for %s", arxiv_id)
        raise HTTPException(
            status_code=500, detail=f"Pattern extraction failed: {exc}"
        ) from exc

    logger.info("Pattern preview generated for %s: %s", arxiv_id, pattern.name)
    return pattern


@app.post("/api/patterns/register", response_model=PatternRegisterResponse)
def register_pattern(
    body: PatternRegisterRequest,
    background_tasks: BackgroundTasks,
) -> PatternRegisterResponse:
    """ユーザーが確認・修正済みの AbstractionPattern を正式に登録する。

    1. Neo4j にパターンノードを保存
    2. Qdrant にパターンの Embedding を保存
    3. 過去論文に対する再評価バッチをバックグラウンドでトリガー
    """
    pattern = body.pattern

    # 1. Neo4j にパターンノードを保存
    try:
        driver = get_driver()
        with driver.session() as session:
            session.run(
                """
                MERGE (p:Paper {arxiv_id: $source_arxiv_id})
                CREATE (ap:AbstractionPattern {
                    pattern_id:         $pattern_id,
                    name:               $name,
                    description:        $description,
                    variables_template: $variables_template,
                    structural_rules:   $structural_rules,
                    source_arxiv_id:    $source_arxiv_id
                })
                CREATE (p)-[:HAS_PATTERN]->(ap)
                """,
                pattern_id=pattern.pattern_id,
                name=pattern.name,
                description=pattern.description,
                variables_template=json.dumps(pattern.variables_template),
                structural_rules=json.dumps(pattern.structural_rules),
                source_arxiv_id=pattern.source_arxiv_id,
            )
        logger.info("Pattern %s registered in Neo4j", pattern.pattern_id)
    except Exception as exc:
        logger.exception("Failed to register pattern %s in Neo4j", pattern.pattern_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to register pattern: {exc}"
        ) from exc

    # 2. Qdrant にパターンの Embedding を保存
    try:
        client = get_client()
        settings = get_settings()
        rules_text = "; ".join(pattern.structural_rules) if pattern.structural_rules else ""
        pattern_text = f"{pattern.name}. {pattern.description} Rules: {rules_text}"
        embed_and_store_pattern(
            pattern_id=pattern.pattern_id,
            pattern_text=pattern_text,
            openai_client=client,
            embedding_model=settings.embedding_model,
        )
    except Exception:
        logger.warning(
            "Pattern embedding failed for %s (continuing)", pattern.pattern_id, exc_info=True
        )

    # 3. バックグラウンドで過去論文に対する再評価バッチをトリガー
    background_tasks.add_task(
        _run_pattern_evaluation_task, pattern
    )

    logger.info("Pattern %s fully registered, batch evaluation queued", pattern.pattern_id)
    return PatternRegisterResponse(pattern_id=pattern.pattern_id, status="registered")


@app.get("/api/patterns", response_model=PatternListResponse)
def list_patterns() -> PatternListResponse:
    """登録済みの全パターン一覧を Neo4j から取得する。"""
    driver = get_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (ap:AbstractionPattern)
            RETURN ap.pattern_id         AS pattern_id,
                   ap.name               AS name,
                   ap.description         AS description,
                   ap.variables_template  AS variables_template,
                   ap.structural_rules    AS structural_rules,
                   ap.source_arxiv_id     AS source_arxiv_id
            ORDER BY ap.name
            """
        ).data()

    patterns = []
    for r in records:
        # variables_template と structural_rules は JSON 文字列で保存されている
        vt = r.get("variables_template", "[]")
        sr = r.get("structural_rules", "[]")
        try:
            vt_parsed = json.loads(vt) if isinstance(vt, str) else vt
        except Exception:
            vt_parsed = []
        try:
            sr_parsed = json.loads(sr) if isinstance(sr, str) else sr
        except Exception:
            sr_parsed = []

        patterns.append(
            PatternOut(
                pattern_id=r["pattern_id"],
                name=r.get("name", ""),
                description=r.get("description", ""),
                variables_template=vt_parsed,
                structural_rules=sr_parsed,
                source_arxiv_id=r.get("source_arxiv_id", ""),
            )
        )

    return PatternListResponse(patterns=patterns)


@app.get("/api/papers/{arxiv_id}/patterns", response_model=PaperPatternListResponse)
def get_paper_patterns(arxiv_id: str) -> PaperPatternListResponse:
    """指定論文にマッチしたパターン一覧を Neo4j から取得する。

    HAS_PATTERN（元論文 → パターン）と MATCHES_PATTERN（マッチ先論文 → パターン）
    の両方を検索する。
    """
    driver = get_driver()
    with driver.session() as session:
        records = session.run(
            """
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[r:MATCHES_PATTERN]->(ap:AbstractionPattern)
            RETURN r.match_id             AS match_id,
                   ap.pattern_id          AS pattern_id,
                   ap.name                AS pattern_name,
                   $arxiv_id              AS target_arxiv_id,
                   r.mapping_explanation  AS mapping_explanation,
                   r.confidence_score     AS confidence_score
            UNION
            MATCH (p:Paper {arxiv_id: $arxiv_id})-[:HAS_PATTERN]->(ap:AbstractionPattern)
            RETURN ap.pattern_id          AS match_id,
                   ap.pattern_id          AS pattern_id,
                   ap.name                AS pattern_name,
                   $arxiv_id              AS target_arxiv_id,
                   'Source paper'         AS mapping_explanation,
                   1.0                    AS confidence_score
            """,
            arxiv_id=arxiv_id,
        ).data()

    matches = [
        PatternMatchOut(
            match_id=r.get("match_id", ""),
            pattern_id=r.get("pattern_id", ""),
            pattern_name=r.get("pattern_name"),
            target_arxiv_id=r.get("target_arxiv_id", ""),
            mapping_explanation=r.get("mapping_explanation", ""),
            confidence_score=float(r.get("confidence_score", 0.0)),
        )
        for r in records
    ]

    return PaperPatternListResponse(matches=matches)


# ---------------------------------------------------------------------------
# Missing Link Suggestion endpoint
# ---------------------------------------------------------------------------

@app.get(
    "/api/patterns/{pattern_id}/suggestions",
    response_model=MissingLinkSuggestionResponse,
)
def get_pattern_suggestions(
    pattern_id: str,
    refresh: bool = Query(False, description="Force re-generation ignoring cache"),
) -> MissingLinkSuggestionResponse:
    """パターンの構造的空白を検知し、異分野への検索サジェストを生成する。

    結果は Neo4j にキャッシュし、再呼び出し時はキャッシュを返す。
    ``refresh=true`` でキャッシュを無視して再生成する。
    """

    driver = get_driver()

    # ── 1. Neo4j からパターンメタデータを取得 ──────────────────────────────
    with driver.session() as session:
        record = session.run(
            """
            MATCH (ap:AbstractionPattern {pattern_id: $pid})
            RETURN ap.name               AS name,
                   ap.description         AS description,
                   ap.variables_template  AS variables_template,
                   ap.structural_rules    AS structural_rules,
                   ap.source_arxiv_id     AS source_arxiv_id
            """,
            pid=pattern_id,
        ).single()

    if not record:
        raise HTTPException(status_code=404, detail=f"Pattern '{pattern_id}' not found")

    pat_name = record.get("name", "")
    pat_desc = record.get("description", "")
    vt_raw = record.get("variables_template", "[]")
    sr_raw = record.get("structural_rules", "[]")

    try:
        variables_template = json.loads(vt_raw) if isinstance(vt_raw, str) else (vt_raw or [])
    except Exception:
        variables_template = []
    try:
        structural_rules = json.loads(sr_raw) if isinstance(sr_raw, str) else (sr_raw or [])
    except Exception:
        structural_rules = []

    # ── 2. キャッシュ確認（refresh=false のとき） ─────────────────────────
    if not refresh:
        with driver.session() as session:
            cached = session.run(
                """
                MATCH (ap:AbstractionPattern {pattern_id: $pid})
                      -[:HAS_SUGGESTIONS]->(s:MissingLinkSuggestion)
                RETURN s.suggestions_json AS suggestions_json
                """,
                pid=pattern_id,
            ).single()
        if cached and cached.get("suggestions_json"):
            try:
                cached_data = json.loads(cached["suggestions_json"])
                return MissingLinkSuggestionResponse(
                    pattern_id=pattern_id,
                    pattern_name=pat_name,
                    suggestions=[SuggestionOut(**s) for s in cached_data],
                    cached=True,
                )
            except Exception:
                logger.warning("Failed to parse cached suggestions for %s, regenerating", pattern_id)

    # ── 3. 既にカバーされている分野を収集 ─────────────────────────────────
    existing_fields: list[str] = []
    with driver.session() as session:
        matched_records = session.run(
            """
            MATCH (p:Paper)-[:MATCHES_PATTERN|HAS_PATTERN]->
                  (ap:AbstractionPattern {pattern_id: $pid})
            RETURN DISTINCT p.categories AS categories
            """,
            pid=pattern_id,
        ).data()
    for mr in matched_records:
        cats = mr.get("categories")
        if cats:
            if isinstance(cats, str):
                try:
                    cats = json.loads(cats)
                except Exception:
                    cats = [cats]
            if isinstance(cats, list):
                existing_fields.extend(cats)

    # ── 4. LLM で Missing Link サジェストを生成 ──────────────────────────
    try:
        result = generate_missing_link_suggestions(
            pattern_name=pat_name,
            pattern_description=pat_desc,
            structural_rules=structural_rules,
            variables_template=variables_template,
            existing_fields=list(set(existing_fields)) if existing_fields else None,
        )
    except Exception as exc:
        logger.exception("Missing link suggestion failed for %s", pattern_id)
        raise HTTPException(
            status_code=500,
            detail=f"Suggestion generation failed: {exc}",
        ) from exc

    suggestions = result.get("suggestions", [])

    # ── 5. Neo4j にキャッシュ保存 ────────────────────────────────────────
    try:
        with driver.session() as session:
            session.run(
                """
                MATCH (ap:AbstractionPattern {pattern_id: $pid})
                OPTIONAL MATCH (ap)-[r:HAS_SUGGESTIONS]->(old:MissingLinkSuggestion)
                DELETE r, old
                WITH ap
                CREATE (s:MissingLinkSuggestion {
                    suggestions_json: $sj,
                    created_at: datetime()
                })
                CREATE (ap)-[:HAS_SUGGESTIONS]->(s)
                """,
                pid=pattern_id,
                sj=json.dumps(suggestions, ensure_ascii=False),
            )
    except Exception:
        logger.warning("Failed to cache suggestions for %s", pattern_id, exc_info=True)

    return MissingLinkSuggestionResponse(
        pattern_id=pattern_id,
        pattern_name=pat_name,
        suggestions=[SuggestionOut(**s) for s in suggestions],
        cached=False,
    )
