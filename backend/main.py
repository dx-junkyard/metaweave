"""MetaWeave v1 — FastAPI backend.

Endpoints
---------
GET  /api/search                        Search arXiv for papers.
POST /api/fetch                         Download a paper PDF and store it in MinIO.
POST /api/extract                       Submit async background extraction job.
GET  /api/extract-status/{arxiv_id}     Poll extraction job status (pending/processing/completed/failed).
GET  /api/extract-result/{arxiv_id}     Fetch a previously extracted paper structure from MinIO.
PUT  /api/extract-result/{arxiv_id}     Update an extracted paper structure in MinIO.
GET  /api/presigned-url                 Return a browser-accessible pre-signed URL for a stored PDF.
GET  /api/papers                        List all stored paper object names.
POST /api/propose-structure             Submit a user structure proposal (saved to Neo4j, async LLM review).
POST /api/auth/register                 Register a new user (Neo4j User node, returns JWT).
POST /api/auth/login                    Authenticate and return a JWT.
GET  /api/auth/me                       Return the current user's profile (requires Bearer token).
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
from metaweave.chat import generate_chat_response
from metaweave.db import get_driver
from metaweave.harvester import PaperMeta, fetch_and_store, search_arxiv
from metaweave.schema import PaperStructure, StructureProposal
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

def _run_extraction_task(object_name: str, arxiv_id: str) -> None:
    """バックグラウンドで実行される抽出タスク。

    1. ステータスを "processing" に更新
    2. MinIO から PDF を取得
    3. テキスト抽出 → 仮説検証型チャンク解析 → Embedding（並行）
    4. 結果を MinIO に保存
    5. ステータスを "completed" または "failed" に更新
    """
    with _job_lock:
        _job_status[arxiv_id] = ExtractJobStatus(arxiv_id=arxiv_id, status="processing")

    try:
        # 1. Fetch PDF from MinIO
        logger.info("Background extraction started for %s", arxiv_id)
        response = _storage().client.get_object("raw-papers", object_name)
        pdf_bytes = response.read()
        response.close()
        response.release_conn()

        # 2. Extract text and structure (includes concurrent embedding)
        text = ext.extract_text_from_pdf_bytes(pdf_bytes)
        structure = ext.extract_paper_structure(text, paper_id=arxiv_id)

        # 3. Persist JSON to extracted-structures bucket
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
    """
    with _job_lock:
        _job_status[body.arxiv_id] = ExtractJobStatus(
            arxiv_id=body.arxiv_id, status="pending"
        )
    background_tasks.add_task(_run_extraction_task, body.object_name, body.arxiv_id)
    logger.info("Extraction job queued for %s", body.arxiv_id)
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
