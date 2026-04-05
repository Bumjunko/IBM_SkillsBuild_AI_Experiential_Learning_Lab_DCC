"""Tests for N5: Retrieval node."""

from commcopilot.nodes.retrieval import retrieve_history


class TestRetrieveHistory:
    def test_returns_empty_list(self, base_state):
        result = retrieve_history(base_state)
        assert result["relevant_history"] == []

    def test_empty_transcript(self, base_state):
        base_state["transcript"] = ""
        result = retrieve_history(base_state)
        assert result["relevant_history"] == []
