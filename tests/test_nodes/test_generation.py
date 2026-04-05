"""Tests for N6: Phrase Generation node."""

import json
from unittest.mock import patch, MagicMock
import pytest
from commcopilot.nodes.generation import generate_phrases
from commcopilot.config import FALLBACK_PHRASES


def _mock_model(response_text):
    mock = MagicMock()
    mock.generate_text.return_value = response_text
    return mock


class TestGeneratePhrases:
    @patch("commcopilot.nodes.generation.ModelInference")
    @patch("commcopilot.nodes.generation.Credentials")
    def test_happy_path(self, mock_creds, mock_model_cls, base_state):
        phrases = ["Could you clarify the deadline?", "Is it due this Friday?", "I'll submit it on time."]
        mock_model_cls.return_value = _mock_model(json.dumps(phrases))
        base_state["context_role"] = "professor"
        base_state["context_tone"] = "formal"
        base_state["context_intent"] = "ask_clarification"

        result = generate_phrases(base_state)
        assert len(result["phrases"]) == 3
        assert result["error"] is None

    @patch("commcopilot.nodes.generation.ModelInference")
    @patch("commcopilot.nodes.generation.Credentials")
    def test_malformed_json_returns_fallback(self, mock_creds, mock_model_cls, base_state):
        mock_model_cls.return_value = _mock_model("not json at all")

        result = generate_phrases(base_state)
        assert result["phrases"] == list(FALLBACK_PHRASES)
        assert "generation_json_error" in result["error"]

    @patch("commcopilot.nodes.generation.ModelInference")
    @patch("commcopilot.nodes.generation.Credentials")
    def test_api_failure_returns_fallback(self, mock_creds, mock_model_cls, base_state):
        mock_instance = MagicMock()
        mock_instance.generate_text.side_effect = Exception("API down")
        mock_model_cls.return_value = mock_instance

        result = generate_phrases(base_state)
        assert result["phrases"] == list(FALLBACK_PHRASES)
        assert "generation_error" in result["error"]

    @patch("commcopilot.nodes.generation.ModelInference")
    @patch("commcopilot.nodes.generation.Credentials")
    def test_empty_list_returns_fallback(self, mock_creds, mock_model_cls, base_state):
        mock_model_cls.return_value = _mock_model("[]")

        result = generate_phrases(base_state)
        assert result["phrases"] == list(FALLBACK_PHRASES)

    @patch("commcopilot.nodes.generation.ModelInference")
    @patch("commcopilot.nodes.generation.Credentials")
    def test_with_relevant_history(self, mock_creds, mock_model_cls, base_state):
        phrases = ["Thank you, professor.", "I understand.", "I'll check the portal."]
        mock_model_cls.return_value = _mock_model(json.dumps(phrases))
        base_state["relevant_history"] = ["Previously discussed assignment submission process"]

        result = generate_phrases(base_state)
        assert len(result["phrases"]) == 3
