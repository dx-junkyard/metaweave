"""MetaWeave v1 — FastAPI backend.

Endpoints
---------
GET  /api/search        Search arXiv for papers.
POST /api/fetch         Download a paper PDF and store it in MinIO.
POST /api/extract       Extract paper structure from a stored PDF using LLM.
GET  /api/presigned-url Return a browser-accessible pre-signed URL for a stored PDF.
GET  /api/papers        List all stored paper object names.
GET  /healthz           Health check.
"""

from __future__ import annotations

import io
import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from metaweave import extractor as ext
from metaweave.harvester import PaperMeta, fetch_and_store, search_arxiv
from metaweave.schema import PaperStructure
from metaweave.storage import StorageManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MetaWeave API", version="0.1.0")

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


@app.post("/api/extract", response_model=PaperStructure)
def extract(body: ExtractRequest) -> PaperStructure:
    """Extract paper structure from a stored PDF using the configured LLM.

    1. Fetch the PDF bytes from MinIO (raw-papers bucket).
    2. Extract plain text with PyMuPDF.
    3. Call the OpenAI Structured Outputs API to obtain a PaperStructure.
    4. Persist the result as JSON in the extracted-structures bucket.
    """
    # 1. Fetch PDF from MinIO
    try:
        response = _storage().client.get_object("raw-papers", body.object_name)
        pdf_bytes = response.read()
        response.close()
        response.release_conn()
    except Exception as exc:
        logger.exception("Failed to fetch PDF from MinIO")
        raise HTTPException(status_code=502, detail=f"MinIO fetch failed: {exc}") from exc

    # 2 & 3. Extract text then structure
    try:
        text = ext.extract_text_from_pdf_bytes(pdf_bytes)
        structure = ext.extract_paper_structure(text, paper_id=body.arxiv_id)
    except EnvironmentError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Structure extraction failed")
        raise HTTPException(status_code=502, detail=f"Extraction failed: {exc}") from exc

    # 4. Persist JSON to extracted-structures bucket (best-effort)
    try:
        json_bytes = structure.model_dump_json().encode()
        safe_id = body.arxiv_id.replace("/", "_")
        _storage().client.put_object(
            "extracted-structures",
            f"{safe_id}.json",
            io.BytesIO(json_bytes),
            length=len(json_bytes),
            content_type="application/json",
        )
    except Exception:
        logger.warning("Could not persist extracted structure to MinIO", exc_info=True)

    return structure


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
