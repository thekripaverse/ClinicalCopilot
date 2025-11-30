from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class TriggerWorkflowRequest(BaseModel):
    patient_id: str
    # In real version this would be audio – for now we simulate with a text note
    note_text: str


class TriggerWorkflowResponse(BaseModel):
    state: Dict[str, Any]


class HumanReviewRequest(BaseModel):
    patient_id: str
    approved: bool
    doctor_comments: Optional[str] = None


class EMRUpdatePayload(BaseModel):
    patient_id: str
    note_summary: str
    prescription: str
    ordered_tests: List[str]
