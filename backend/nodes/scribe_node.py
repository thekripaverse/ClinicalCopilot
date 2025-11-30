from ..state import AgentState


def scribe_node(state: AgentState) -> AgentState:
    """
    Converts raw note_text into transcript + simple summary.
    For now, we assume frontend sends note_text directly.
    """
    # In future: call STT tool_transcribe_voice on audio
    state.raw_transcript = state.raw_transcript or state.note_summary or ""
    if not state.note_summary and state.raw_transcript:
        # naive summary
        state.note_summary = state.raw_transcript[:250]
    state.audit_log.append("Scribe node: captured note summary.")
    return state
