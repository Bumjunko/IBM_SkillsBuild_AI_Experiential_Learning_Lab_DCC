"""LangGraph StateGraph definition for CommCopilot pipeline.

Graph structure:
    trigger -> [N2 Embedding, N3 Context] (parallel fan-out)
            -> N5 Retrieval (conditional: confidence < threshold)
            -> N6 Generation
            -> N7 Safety Filter
            -> output
"""

from langgraph.graph import StateGraph, START, END
from commcopilot.state import PipelineState
from commcopilot.config import CONFIDENCE_THRESHOLD, FALLBACK_PHRASES
from commcopilot.nodes.embedding import embed_transcript
from commcopilot.nodes.context import infer_context
from commcopilot.nodes.retrieval import retrieve_history
from commcopilot.nodes.generation import generate_phrases
from commcopilot.nodes.safety import filter_phrases


def _route_after_context(state: PipelineState) -> str:
    """Route to retrieval if confidence is low, otherwise skip to generation."""
    if state.get("error") and "context" in state.get("error", ""):
        return "generate"
    if state.get("context_confidence", 1.0) < CONFIDENCE_THRESHOLD:
        return "retrieve"
    return "generate"


def _route_after_safety(state: PipelineState) -> str:
    """Always end after safety filter."""
    return END


def build_graph() -> StateGraph:
    """Build and compile the CommCopilot LangGraph pipeline."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("embed", embed_transcript)
    graph.add_node("context", infer_context)
    graph.add_node("retrieve", retrieve_history)
    graph.add_node("generate", generate_phrases)
    graph.add_node("safety", filter_phrases)

    # Entry: fan-out to embed + context in parallel
    graph.add_edge(START, "embed")
    graph.add_edge(START, "context")

    # N2 embed is fire-and-forget, goes to END
    graph.add_edge("embed", END)

    # N3 context -> conditional: retrieve or generate
    graph.add_conditional_edges("context", _route_after_context, {
        "retrieve": "retrieve",
        "generate": "generate",
    })

    # N5 retrieve -> N6 generate
    graph.add_edge("retrieve", "generate")

    # N6 generate -> N7 safety
    graph.add_edge("generate", "safety")

    # N7 safety -> END
    graph.add_edge("safety", END)

    return graph.compile()


# Pre-built compiled graph instance
pipeline = build_graph()
