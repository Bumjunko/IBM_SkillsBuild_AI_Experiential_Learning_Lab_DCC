"""Shared test fixtures for CommCopilot tests."""

import pytest
from commcopilot.state import PipelineState


@pytest.fixture
def base_state() -> PipelineState:
    """A minimal valid pipeline state for testing."""
    return {
        "session_id": "test-session-123",
        "scenario": "office_hours",
        "transcript": "The deadline is Friday. Submit via portal. Um, what about the rubric?",
        "session_history": [],
        "hesitation_trigger": "pause",
        "embedding_stored": False,
        "context_role": "",
        "context_tone": "",
        "context_formality": "",
        "context_intent": "",
        "context_confidence": 0.0,
        "relevant_history": [],
        "phrases": [],
        "safe_phrases": [],
        "error": None,
    }


@pytest.fixture
def mock_granite_response():
    """Factory for mocking Granite model responses."""
    def _make(response_text):
        class MockModel:
            def generate_text(self, prompt):
                return response_text
        return MockModel()
    return _make
