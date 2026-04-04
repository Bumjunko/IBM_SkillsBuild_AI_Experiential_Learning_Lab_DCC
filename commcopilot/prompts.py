"""Granite prompt templates for CommCopilot pipeline nodes."""


def context_inference_prompt(transcript: str, scenario_context: str) -> str:
    return f"""You are analyzing a live conversation between an international student and another person.
The scenario: {scenario_context}

The transcript below contains speech from BOTH speakers (student and the other person).
Identify which parts are likely the student (hesitant, shorter responses, possible filler words)
and which parts are the other person (more confident, directive, longer statements).

Analyze the conversation and return a JSON object with these fields:
- role: the other person's role (e.g., "professor", "admin_staff", "advisor")
- tone: the current tone of conversation ("formal", "semi-formal", "informal")
- formality: level of formality expected in the student's response ("high", "medium", "low")
- intent: what the student likely needs to say next (e.g., "ask_clarification", "confirm", "respond_to_question", "express_opinion")
- confidence: your confidence in this analysis, from 0.0 to 1.0

Transcript:
{transcript}

Return ONLY valid JSON, no other text:"""


def phrase_generation_prompt(
    transcript: str,
    context_role: str,
    context_tone: str,
    context_intent: str,
    relevant_history: list[str],
    scenario_context: str,
) -> str:
    history_section = ""
    if relevant_history:
        history_text = "\n".join(f"- {h}" for h in relevant_history)
        history_section = f"\nRelevant past conversation context:\n{history_text}\n"

    return f"""You are helping an international student who just hesitated during a live English conversation.
The scenario: {scenario_context}

Current conversation context:
- Speaking with: {context_role}
- Tone: {context_tone}
- The student likely needs to: {context_intent}
{history_section}
Recent transcript:
{transcript}

Generate exactly 3 short, natural English phrases the student could say right now.
The phrases should:
- Be contextually appropriate for the situation
- Match the expected formality level
- Be concise (under 15 words each)
- Sound natural, not robotic
- Be safe and respectful

Return ONLY a JSON array of 3 strings, no other text:"""


def safety_filter_prompt(phrases: list[str]) -> str:
    phrases_text = "\n".join(f'{i+1}. "{p}"' for i, p in enumerate(phrases))
    return f"""Review these phrase suggestions for an international student in a live conversation.

Phrases to review:
{phrases_text}

Reject any phrase that contains:
- Profanity, slurs, or offensive language
- Culturally demeaning or insensitive framing
- Medically or legally inaccurate advice
- Manipulative or aggressive language
- Anything inappropriate for an academic/professional setting

Return a JSON object with:
- safe: array of phrases that passed the review
- rejected: array of phrases that were rejected, with reason

Return ONLY valid JSON, no other text:"""


def recap_prompt(transcript: str, phrases_used: list[str]) -> str:
    used_text = "\n".join(f"- {p}" for p in phrases_used) if phrases_used else "- None"
    return f"""Summarize this conversation session for the student.

Full transcript:
{transcript}

Phrases the student used from suggestions:
{used_text}

Provide a brief, encouraging recap (3-5 sentences) covering:
1. What the conversation was about
2. How the student handled it
3. One specific tip for next time

Keep it supportive and constructive. Return plain text, no JSON:"""
