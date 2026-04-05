"""Tests for N2: Embedding node."""

from commcopilot.nodes.embedding import embed_transcript


class TestEmbedTranscript:
    def test_happy_path(self, base_state):
        result = embed_transcript(base_state)
        assert result["embedding_stored"] is True

    def test_empty_transcript_skips(self, base_state):
        base_state["transcript"] = ""
        result = embed_transcript(base_state)
        assert result["embedding_stored"] is False

    def test_whitespace_only_skips(self, base_state):
        base_state["transcript"] = "   "
        result = embed_transcript(base_state)
        assert result["embedding_stored"] is False
