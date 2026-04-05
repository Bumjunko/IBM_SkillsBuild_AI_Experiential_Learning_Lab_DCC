"""Tests for N1: STT node."""

from commcopilot.nodes.stt import update_transcript


class TestUpdateTranscript:
    def test_updates_history(self, base_state):
        result = update_transcript(base_state)
        assert len(result["session_history"]) == 1
        assert base_state["transcript"] in result["session_history"][0]

    def test_empty_transcript(self, base_state):
        base_state["transcript"] = ""
        result = update_transcript(base_state)
        assert result["session_history"] == []

    def test_whitespace_only(self, base_state):
        base_state["transcript"] = "   "
        result = update_transcript(base_state)
        assert result["session_history"] == []
