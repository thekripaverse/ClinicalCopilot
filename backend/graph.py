from .state import AgentState
from .nodes.scribe_node import scribe_node
from .nodes.planner_node import planner_node
from .nodes.rx_node import rx_node
from .nodes.safety_node import safety_node
from .nodes.hil_node import hil_wait_node
from .nodes.symptom_node import symptom_node 
from .tools import tool_update_emr 

def run_initial_workflow(state: AgentState) -> AgentState:
    """
    Runs the automatic portion of the workflow until human review is needed.
    """
    state.audit_log.append("Workflow: starting initial pipeline.")
    state = scribe_node(state)
    state = planner_node(state)
    state = rx_node(state)
    state = safety_node(state)
    state = symptom_node(state)

    if state.requires_review:
        state = hil_wait_node(state)
        state.audit_log.append("Workflow: paused for human-in-the-loop.")
    else:
        state.audit_log.append("Workflow: completed without human review.")
    try:
        payload = {
            "patient_id": state.patient_id,
            "note_summary": state.note_summary,
            "symptoms": state.symptoms,
            "suggested_tests": state.suggested_tests,
            "draft_prescription": state.draft_prescription,
            "safety_flags": state.safety_flags,
        }
        result = tool_update_emr(payload)
        state.executed_actions.append(result)
        state.audit_log.append("Workflow: EMR update stored.")
    except Exception as e:
        state.audit_log.append(f"Workflow: EMR update failed: {e!r}")
    return state
