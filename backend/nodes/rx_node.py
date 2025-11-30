from ..state import AgentState


def rx_node(state: AgentState) -> AgentState:
    """
    Drafts a simple, rule-based 'prescription' from symptoms.
    In reality, this would be LLM + guardrails + human review.
    """
    summary = state.note_summary or state.raw_transcript or ""
    symptoms = ", ".join(state.symptoms) if state.symptoms else "unspecified symptoms"

    draft = (
        f"Provisional prescription for {symptoms}.\n"
        f"Note summary: {summary[:200]}...\n\n"
        "Medications:\n"
        "- (To be decided by physician)\n\n"
        "Instructions:\n"
        "- Follow up if symptoms worsen.\n"
    )

    state.draft_prescription = draft
    state.audit_log.append("Rx node: drafted provisional prescription.")
    return state
