"""Tests for N3: Context Inference node."""

import json
from unittest.mock import patch, MagicMock
import pytest
from commcopilot.nodes.context import infer_context


@pytest.fixture
def good_context_json():
    return json.dumps({
        "role": "professor",
        "tone": "formal",
        "formality": "high",
        "intent": "ask_clarification",
        "confidence": 0.85,
    })


@pytest.fixture
def bad_json():
    return "this is not json {{"


def _mock_model(response_text):
    """Create a mock that patches ModelInference."""
    mock_model = MagicMock()
    mock_model.generate_text.return_value = response_text
    return mock_model


class TestInferContext:
    @patch("commcopilot.nodes.context.ModelInference")
    @patch("commcopilot.nodes.context.Credentials")
    def test_happy_path(self, mock_creds, mock_model_cls, base_state, good_context_json):
        mock_model_cls.return_value = _mock_model(good_context_json)
        result = infer_context(base_state)

        assert result["context_role"] == "professor"
        assert result["context_tone"] == "formal"
        assert result["context_confidence"] == 0.85
        assert result["error"] is None

    @patch("commcopilot.nodes.context.ModelInference")
    @patch("commcopilot.nodes.context.Credentials")
    def test_malformed_json_returns_defaults(self, mock_creds, mock_model_cls, base_state, bad_json):
        mock_model_cls.return_value = _mock_model(bad_json)
        result = infer_context(base_state)

        assert "context_json_error" in result["error"]
        assert result["context_confidence"] == 0.3

    @patch("commcopilot.nodes.context.ModelInference")
    @patch("commcopilot.nodes.context.Credentials")
    def test_api_timeout(self, mock_creds, mock_model_cls, base_state):
        mock_instance = MagicMock()
        mock_instance.generate_text.side_effect = TimeoutError("API timeout")
        mock_model_cls.return_value = mock_instance

        result = infer_context(base_state)
        assert "context_error" in result["error"]
        assert result["context_confidence"] == 0.3

    def test_empty_transcript(self, base_state):
        base_state["transcript"] = ""
        # With empty transcript, it should still attempt (Granite handles empty input)
        # but we're testing it doesn't crash
        with patch("commcopilot.nodes.context.ModelInference") as mock_cls:
            with patch("commcopilot.nodes.context.Credentials"):
                mock_cls.return_value = _mock_model('{"role":"unknown","tone":"neutral","formality":"medium","intent":"unclear","confidence":0.2}')
                result = infer_context(base_state)
                assert result["context_confidence"] == 0.2
