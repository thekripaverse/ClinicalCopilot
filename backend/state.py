from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """Shared state that flows through agents in the workflow."""

    patient_id: Optional[str] = None
    raw_transcript: Optional[str] = None       # Output of STT / scribe
    note_summary: Optional[str] = None         # Structured, short summary
    symptoms: List[str] = Field(default_factory=list)
    suggested_tests: List[str] = Field(default_factory=list)
    guideline_hits: List[Dict[str, Any]] = Field(default_factory=list)

    draft_prescription: Optional[str] = None   # Rx agent output
    safety_flags: List[str] = Field(default_factory=list)
    requires_review: bool = False

    actions_to_execute: List[Dict[str, Any]] = Field(default_factory=list)
    executed_actions: List[Dict[str, Any]] = Field(default_factory=list)

    audit_log: List[str] = Field(default_factory=list)
