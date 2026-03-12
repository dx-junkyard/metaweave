"""Tests for the Semantic Scholar citation count integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from metaweave.citations import fetch_citation_counts


class TestFetchCitationCounts:
    """Tests for fetch_citation_counts()."""

    def test_empty_list_returns_empty_dict(self):
        result = fetch_citation_counts([])
        assert result == {}

    @patch("metaweave.citations._post_batch")
    def test_successful_batch_response(self, mock_post: MagicMock):
        mock_post.return_value = {
            "2301.00001": 42,
            "2301.00002": 7,
        }
        result = fetch_citation_counts(["2301.00001", "2301.00002"])
        assert result == {"2301.00001": 42, "2301.00002": 7}
        mock_post.assert_called_once_with(["2301.00001", "2301.00002"])

    @patch("metaweave.citations._post_batch")
    def test_missing_paper_defaults_to_absent(self, mock_post: MagicMock):
        """Papers not found in Semantic Scholar are simply absent from the result dict."""
        mock_post.return_value = {"2301.00001": 10}
        result = fetch_citation_counts(["2301.00001", "2301.00002"])
        assert result == {"2301.00001": 10}
        # The caller should use .get(id, 0) to default missing ones to 0.

    @patch("metaweave.citations._post_batch")
    def test_batch_failure_returns_empty(self, mock_post: MagicMock):
        """If the API call fails after retries, gracefully return empty."""
        mock_post.side_effect = Exception("API error")
        result = fetch_citation_counts(["2301.00001"])
        assert result == {}

    @patch("metaweave.citations._post_batch")
    def test_large_batch_split(self, mock_post: MagicMock):
        """Lists > 500 papers should be split into multiple batches."""
        ids = [f"2301.{i:05d}" for i in range(600)]
        mock_post.return_value = {}
        fetch_citation_counts(ids)
        assert mock_post.call_count == 2
        assert len(mock_post.call_args_list[0][0][0]) == 500
        assert len(mock_post.call_args_list[1][0][0]) == 100


class TestCitationMergeLogic:
    """Tests for citation count merge and sort logic."""

    @patch("metaweave.citations._post_batch")
    def test_merge_and_sort_by_citation(self, mock_post: MagicMock):
        """Simulate the merge + sort logic used in the search endpoint."""
        mock_post.return_value = {"2301.00001": 5, "2301.00002": 100}
        citation_map = fetch_citation_counts(["2301.00001", "2301.00002"])

        # Simulate sorting
        items = [
            {"arxiv_id": "2301.00001", "count": citation_map.get("2301.00001", 0)},
            {"arxiv_id": "2301.00002", "count": citation_map.get("2301.00002", 0)},
        ]
        items.sort(key=lambda m: m["count"], reverse=True)

        assert items[0]["arxiv_id"] == "2301.00002"
        assert items[0]["count"] == 100
        assert items[1]["arxiv_id"] == "2301.00001"
        assert items[1]["count"] == 5

    @patch("metaweave.citations._post_batch")
    def test_missing_papers_default_to_zero(self, mock_post: MagicMock):
        """Papers not in Semantic Scholar should default to 0 via .get()."""
        mock_post.return_value = {"2301.00001": 10}
        citation_map = fetch_citation_counts(["2301.00001", "2301.00002"])

        assert citation_map.get("2301.00001", 0) == 10
        assert citation_map.get("2301.00002", 0) == 0
