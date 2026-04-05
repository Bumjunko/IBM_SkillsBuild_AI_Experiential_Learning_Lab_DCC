"""Tests for N7: Safety Filter node."""

import json
from unittest.mock import patch, MagicMock
import pytest
from commcopilot.nodes.safety import filter_phrases
from commcopilot.config import FALLBACK_PHRASES


def _mock_model(response_text):
    mock = MagicMock()
    mock.generate_text.return_value = response_text
    return mock


class TestFilterPhrases:
    @patch("commcopilot.nodes.safety.ModelInference")
    @patch("commcopilot.nodes.safety.Credentials")
    def test_all_pass(self, mock_creds, mock_model_cls, base_state):
        safe_response = json.dumps({
            "safe": ["Could you clarify?", "I understand.", "Thank you."],
            "rejected": [],
        })
        mock_model_cls.return_value = _mock_model(safe_response)
        base_state["phrases"] = ["Could you clarify?", "I understand.", "Thank you."]

        result = filter_phrases(base_state)
        assert len(result["safe_phrases"]) == 3
        assert result["error"] is None

    @patch("commcopilot.nodes.safety.ModelInference")
    @patch("commcopilot.nodes.safety.Credentials")
    def test_one_rejected(self, mock_creds, mock_model_cls, base_state):
        safe_response = json.dumps({
            "safe": ["Could you clarify?", "Thank you."],
            "rejected": [{"phrase": "Bad phrase", "reason": "inappropriate"}],
        })
        mock_model_cls.return_value = _mock_model(safe_response)
        base_state["phrases"] = ["Could you clarify?", "Bad phrase", "Thank you."]

        result = filter_phrases(base_state)
        assert len(result["safe_phrases"]) == 2
        assert result["error"] is None

    @patch("commcopilot.nodes.safety.ModelInference")
    @patch("commcopilot.nodes.safety.Credentials")
    def test_all_rejected_triggers_fallback(self, mock_creds, mock_model_cls, base_state):
        # Both calls return no safe phrases
        reject_response = json.dumps({"safe": [], "rejected": [{"phrase": "x", "reason": "bad"}]})
        mock_model_cls.return_value = _mock_model(reject_response)
        base_state["phrases"] = ["bad1", "bad2", "bad3"]

        result = filter_phrases(base_state)
        assert result["safe_phrases"] == list(FALLBACK_PHRASES)

    @patch("commcopilot.nodes.safety.ModelInference")
    @patch("commcopilot.nodes.safety.Credentials")
    def test_json_error_passes_all(self, mock_creds, mock_model_cls, base_state):
        mock_model_cls.return_value = _mock_model("not json")
        base_state["phrases"] = ["phrase1", "phrase2", "phrase3"]

        result = filter_phrases(base_state)
        assert result["safe_phrases"] == ["phrase1", "phrase2", "phrase3"]
        assert "safety_json_error" in result["error"]

    def test_no_phrases_returns_fallback(self, base_state):
        base_state["phrases"] = []
        result = filter_phrases(base_state)
        assert result["safe_phrases"] == list(FALLBACK_PHRASES)

    @patch("commcopilot.nodes.safety.ModelInference")
    @patch("commcopilot.nodes.safety.Credentials")
    def test_api_failure_passes_all(self, mock_creds, mock_model_cls, base_state):
        mock_instance = MagicMock()
        mock_instance.generate_text.side_effect = Exception("API down")
        mock_model_cls.return_value = mock_instance
        base_state["phrases"] = ["safe phrase 1", "safe phrase 2"]

        result = filter_phrases(base_state)
        assert result["safe_phrases"] == ["safe phrase 1", "safe phrase 2"]
        assert "safety_error" in result["error"]
