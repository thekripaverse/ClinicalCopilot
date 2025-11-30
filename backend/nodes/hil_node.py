from ..state import AgentState


def hil_wait_node(state: AgentState) -> AgentState:
    """
    Marks state as 'waiting for human approval'.
    The workflow will resume when an API call provides that approval.
    """
    state.audit_log.append("HIL node: waiting for physician approval.")
    return state


def hil_apply_decision(state: AgentState, approved: bool, doctor_comments: str | None) -> AgentState:
    """
    Applies the physician's decision to proceed or not.
    """
    if approved:
        state.requires_review = False
        state.audit_log.append("HIL node: physician approved actions.")
    else:
        state.audit_log.append("HIL node: physician rejected or modified actions.")

    if doctor_comments:
        state.audit_log.append(f"Physician comment: {doctor_comments}")

    return state
