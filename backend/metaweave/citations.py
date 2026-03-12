"""Semantic Scholar API integration for citation counts."""

from __future__ import annotations

import logging
from typing import Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

SEMANTIC_SCHOLAR_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
SEMANTIC_SCHOLAR_PAPER_URL = "https://api.semanticscholar.org/graph/v1/paper"

# Maximum papers per batch request (Semantic Scholar limit is 500)
_BATCH_SIZE = 500


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def _post_batch(arxiv_ids: list[str]) -> dict:
    """POST to Semantic Scholar batch endpoint with retry logic.

    Returns a dict mapping arXiv ID -> citation count.
    """
    paper_ids = [f"ArXiv:{aid}" for aid in arxiv_ids]
    resp = requests.post(
        SEMANTIC_SCHOLAR_BATCH_URL,
        json={"ids": paper_ids},
        params={"fields": "citationCount,externalIds"},
        timeout=30,
    )
    resp.raise_for_status()
    results: dict[str, int] = {}
    for item in resp.json():
        if item is None:
            continue
        ext_ids = item.get("externalIds") or {}
        aid = ext_ids.get("ArXiv", "")
        count = item.get("citationCount")
        if aid and count is not None:
            results[aid] = count
    return results


def fetch_citation_counts(arxiv_ids: list[str]) -> dict[str, int]:
    """Fetch citation counts for a list of arXiv IDs from Semantic Scholar.

    Returns a dict mapping arXiv ID -> citation count.
    Papers not found in Semantic Scholar default to 0.
    """
    if not arxiv_ids:
        return {}

    result: dict[str, int] = {}

    # Process in batches
    for i in range(0, len(arxiv_ids), _BATCH_SIZE):
        batch = arxiv_ids[i : i + _BATCH_SIZE]
        try:
            batch_result = _post_batch(batch)
            result.update(batch_result)
        except Exception:
            logger.warning(
                "Semantic Scholar batch request failed for %d papers; "
                "falling back to citation_count=0",
                len(batch),
                exc_info=True,
            )

    return result
