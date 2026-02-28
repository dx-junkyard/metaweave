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
GET  /healthz                           Health check.
"""

from __future__ import annotations

import io
import logging
import threading
from functools import lru_cache

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from metaweave import extractor as ext
from metaweave.harvester import PaperMeta, fetch_and_store, search_arxiv
from metaweave.schema import PaperStructure
from metaweave.storage import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
