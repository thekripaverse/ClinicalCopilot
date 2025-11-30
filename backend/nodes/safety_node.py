from ..state import AgentState


RED_FLAG_KEYWORDS = [
    "severe chest pain",
    "shortness of breath",
    "loss of consciousness",
    "suicidal",
    "stroke",
]


def safety_node(state: AgentState) -> AgentState:
    """
    Very simple safety checker that looks for 'red flag' patterns
    in the note summary or transcript.
    """
    text = (state.note_summary or "") + " " + (state.raw_transcript or "")
    text_lower = text.lower()

    flags = []
    for kw in RED_FLAG_KEYWORDS:
        if kw in text_lower:
            flags.append(f"Red flag detected: {kw}")

    if flags:
        state.safety_flags.extend(flags)
        state.requires_review = True
        state.audit_log.append("Safety node: red flags detected, human review required.")
    else:
        state.audit_log.append("Safety node: no red flags detected.")

    return state
