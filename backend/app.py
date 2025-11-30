import os
from tempfile import NamedTemporaryFile
from .face_biometrics import enroll_from_image_bytes, verify_from_image_bytes
import json
from datetime import datetime, timezone
from .auth import authorize_patient, is_patient_authorized
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
from .state import AgentState
from .schemas import (
    TriggerWorkflowRequest,
    TriggerWorkflowResponse,
    HumanReviewRequest,
    EMRUpdatePayload,
)
from .graph import run_initial_workflow
from .tools import tool_update_emr, tool_transcribe_voice, get_qdrant_client, EMR_STORE_PATH, tool_send_to_pharmacy, PHARMACY_STORE_PATH
from .nodes.hil_node import hil_apply_decision
from .db import (
    init_db,
    SessionLocal,
    get_or_create_patient,
    Patient,
    Encounter,
    LabResult,
    RadiologyReport,
    PharmacyOrder,
    InsuranceProfile,
    PatientDoctorAccess
)

class ApproveEMRRequest(BaseModel):
    patient_id: str
    note_summary: str
    symptoms: List[str]
    suggested_tests: List[str]
    draft_prescription: str
app = FastAPI(title="Agentic AI Healthcare Workflow Assistant")
@app.on_event("startup")
def on_startup():
    init_db()

class PharmacySendRequest(BaseModel):
    patient_id: str
    prescription: str
    emr_record_id: Optional[str] = None
    suggested_tests: List[str] = []
    symptoms: List[str] = []


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/qdrant/collections")
def qdrant_list_collections():
    """
    List all Qdrant collections with basic info.
    Uses the same embedded client instance as your RAG tool.
    """
    client = get_qdrant_client()
    collections = client.get_collections().collections
    out: List[Dict[str, Any]] = []

    for c in collections:
        name = c.name
        try:
            info = client.get_collection(name)
            out.append({
                "name": name,
                "vectors_count": info.vectors_count,
                "status": str(info.status),
            })
        except Exception as e:
            out.append({
                "name": name,
                "vectors_count": None,
                "status": f"error: {e}",
            })
    return {"collections": out}


@app.get("/qdrant/collection/{name}")
def qdrant_view_collection(name: str, limit: int = 20):
    """
    View up to `limit` points from a given collection.
    Shows id, payload keys and a short text preview if present.
    """
    client = get_qdrant_client()

    try:
        points, next_offset = client.scroll(
            collection_name=name,
            limit=limit,
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading collection: {e}")

    out_points: List[Dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        # Try to pick some text field from payload
        text_field = (
            payload.get("text")
            or payload.get("chunk")
            or payload.get("content")
            or ""
        )
        preview = text_field
        if isinstance(preview, str) and len(preview) > 200:
            preview = preview[:200] + "..."

        out_points.append({
            "id": p.id,
            "payload_keys": list(payload.keys()),
            "preview": preview,
            "payload": payload,  # full payload for debugging
        })

    return {
        "collection": name,
        "count": len(out_points),
        "points": out_points,
    }

@app.post("/trigger-workflow", response_model=TriggerWorkflowResponse)
def trigger_workflow(req: TriggerWorkflowRequest):
    init_state = AgentState(
        patient_id=req.patient_id,
        raw_transcript=req.note_text,
        note_summary=req.note_text,
    )
    final_state = run_initial_workflow(init_state)
    return {"state": final_state.model_dump()}

@app.get("/get-emr")
def get_emr(patient_id: str):
    records = []

    if EMR_STORE_PATH.exists():
        try:
            with EMR_STORE_PATH.open("r", encoding="utf-8") as f:
                all_records = json.load(f)
        except Exception:
            all_records = []
    else:
        all_records = []
    for rec in all_records:
        if rec.get("patient_id") == patient_id:
            records.append(rec)
    def _get_ts(r):
        ts = r.get("timestamp_utc") or r.get("timestamp") or ""
        return ts
    records.sort(key=_get_ts, reverse=True)
    return records

@app.post("/patient/grant-access")
def grant_access(patient_id: str, doctor_username: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(404, "No such patient")

        record = (
            db.query(PatientDoctorAccess)
              .filter(
                    PatientDoctorAccess.patient_id == patient.id,
                    PatientDoctorAccess.doctor_username == doctor_username,
              )
              .first()
        )

        if not record:
            record = PatientDoctorAccess(
                patient_id=patient.id,
                doctor_username=doctor_username,
                is_allowed=True
            )
            db.add(record)
        else:
            record.is_allowed = True

        db.commit()
        return {"status": "ok", "message": "Doctor access granted."}
    finally:
        db.close()

@app.post("/patient/revoke-access")
def revoke_access(patient_id: str, doctor_username: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(404, "No such patient")

        record = (
            db.query(PatientDoctorAccess)
              .filter(
                    PatientDoctorAccess.patient_id == patient.id,
                    PatientDoctorAccess.doctor_username == doctor_username,
              )
              .first()
        )

        if record:
            record.is_allowed = False
            db.commit()

        return {"status": "ok", "message": "Doctor access revoked."}
    finally:
        db.close()

@app.get("/get-pharmacy-orders")
def get_pharmacy_orders(patient_id: Optional[str] = None):
    records = []

    if PHARMACY_STORE_PATH.exists():
        try:
            with PHARMACY_STORE_PATH.open("r", encoding="utf-8") as f:
                all_orders = json.load(f)
        except Exception:
            all_orders = []
    else:
        all_orders = []

    if patient_id:
        for rec in all_orders:
            if rec.get("patient_id") == patient_id:
                records.append(rec)
    else:
        records = all_orders
    def _get_ts(r):
        return r.get("timestamp_utc") or ""

    records.sort(key=_get_ts, reverse=True)

    return records

@app.post("/human-review")
def human_review(req: HumanReviewRequest):
    state = AgentState(patient_id=req.patient_id)
    state = hil_apply_decision(state, approved=req.approved, doctor_comments=req.doctor_comments)
    return {"message": "Review applied", "state": state.model_dump()}

@app.post("/emr-update")
def emr_update(payload: EMRUpdatePayload):
    result = tool_update_emr(payload.model_dump())
    return {"result": result}

@app.post("/audio-workflow", response_model=TriggerWorkflowResponse)
async def audio_workflow(patient_id: str, audio: UploadFile = File(...)):
    suffix = ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await audio.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        transcript = tool_transcribe_voice(tmp_path)

        init_state = AgentState(
            patient_id=patient_id,
            raw_transcript=transcript,
            note_summary=transcript,
        )
        final_state = run_initial_workflow(init_state)
        return {"state": final_state.model_dump()}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Agentic Clinical Workflow Copilot</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #0f172a, #020617 60%);
      color: #e5e7eb;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      border-bottom: 1px solid #1f2937;
      background: #020617cc;
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    header h1 {
      font-size: 1.25rem;
      color: #38bdf8;
      margin: 0;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      font-size: 0.8rem;
    }
    .pill input {
      border: none;
      outline: none;
      background: transparent;
      color: #e5e7eb;
      font-size: 0.8rem;
    }
    main {
      padding: 16px 24px 32px;
      max-width: 1200px;
      margin: 0 auto;
    }
    .status-line {
      font-size: 0.8rem;
      margin-bottom: 8px;
      min-height: 1.5em;
    }
    .status-ok { color: #4ade80; }
    .status-warn { color: #fb7185; }
    .status-info { color: #38bdf8; }

    /* Stepper / carousel controls */
    .stepper {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .step-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      border: 1px solid #1f2937;
      background: #020617;
      font-size: 0.75rem;
      opacity: 0.5;
    }
    .step-pill.active {
      border-color: #38bdf8;
      box-shadow: 0 0 0 1px #0ea5e9;
      opacity: 1;
    }
    .step-pill span.index {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 18px;
      height: 18px;
      border-radius: 999px;
      background: #0f172a;
      font-size: 0.7rem;
    }
    .step-nav {
      margin-left: auto;
      display: flex;
      gap: 8px;
      align-items: center;
    }
    .btn {
      border: none;
      border-radius: 999px;
      padding: 6px 12px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: transform 0.08s ease, box-shadow 0.08s ease, background 0.1s ease;
    }
    .btn:active { transform: scale(0.97); }
    .btn-primary {
      background: #06b6d4;
      color: #020617;
      box-shadow: 0 8px 18px rgba(8,145,178,0.7);
    }
    .btn-primary:hover { background: #22d3ee; }
    .btn-ghost {
      background: #020617;
      color: #e5e7eb;
      border: 1px solid #1f2937;
    }
    .btn-danger {
      background: #f97373;
      color: #111827;
    }
    .btn[disabled] {
      opacity: 0.5;
      cursor: not-allowed;
      box-shadow: none;
    }

    .slide {
      display: none;
      animation: fadeIn 0.25s ease-out;
    }
    .slide.active {
      display: block;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px);}
      to { opacity: 1; transform: translateY(0);}
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    .card {
      background: #020617;
      border-radius: 16px;
      border: 1px solid #1f2937;
      padding: 14px 16px;
      box-shadow: 0 18px 35px rgba(0,0,0,0.55);
    }
    .card h2, .card h3 {
      margin: 0 0 6px;
      font-size: 0.95rem;
      color: #e5e7eb;
    }
    .card p {
      margin: 4px 0;
      font-size: 0.8rem;
      color: #9ca3af;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      resize: vertical;
      border-radius: 12px;
      border: 1px solid #1f2937;
      background: #020617;
      color: #e5e7eb;
      padding: 8px;
      font-size: 0.8rem;
      outline: none;
    }
    textarea:focus {
      border-color: #38bdf8;
    }
    pre {
      white-space: pre-wrap;
      font-size: 0.75rem;
      max-height: 180px;
      overflow-y: auto;
    }
    ul {
      padding-left: 18px;
      margin: 6px 0;
      font-size: 0.8rem;
    }
    .badge {
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 0.7rem;
      background: #0f172a;
      border: 1px solid #1f2937;
      margin: 2px 4px 2px 0;
    }
    video {
      width: 260px;
      height: 180px;
      border-radius: 16px;
      border: 1px solid #1f2937;
      background: #020617;
      object-fit: cover;
    }
    .emr-item {
      border-radius: 12px;
      border: 1px solid #1f2937;
      padding: 8px 10px;
      margin-bottom: 6px;
      background: #020617;
    }
    code {
      background: #020617;
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #93c5fd;
    }
    .login-modal {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.92);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 50;
    }

    .login-card {
      background: #020617;
      border-radius: 1rem;
      padding: 20px 22px;
      width: 320px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
      border: 1px solid #1f2937;
    }

    .login-card h2 {
      margin: 0 0 8px 0;
      font-size: 1.1rem;
    }

    .login-card label {
      display: flex;
      flex-direction: column;
      gap: 4px;
      margin-bottom: 8px;
      font-size: 0.75rem;
      color: #9ca3af;
    }
    #ehrDemoBox {
      font-size: 0.8rem;
      color: #e5e7eb;
      line-height: 1.4;
    }

    .ehr-label {
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #9ca3af;
      margin-top: 4px;
      margin-bottom: 2px;
    }

    .login-card input,
    .login-card select {
      padding: 6px 10px;
      border-radius: 999px;
      border: none;
      outline: none;
      background: #020617;
      color: #e5e7eb;
      font-size: 0.8rem;
    }

    @media (max-width: 768px) {
      header { flex-direction: column; align-items: flex-start; gap: 8px; }
      .step-nav { margin-left: 0; margin-top: 6px; }
    }
  </style>
</head>
<body>
<header>
  <h1>Agentic Clinical Workflow Copilot</h1>

  <div class="pill">
    <span style="color:#9ca3af;font-size:0.75rem;">Patient ID:</span>
    <input id="patientIdInput" value="P001" />
  </div>

  <div class="pill" id="currentUserInfo"
       style="margin-left:auto;font-size:0.75rem;color:#9ca3af;display:none;">
    Not logged in
  </div>
    <button id="btnLogout" class="btn btn-ghost" style="position:fixed;top:16px;right:16px;font-size:0.7rem;display:none;">
    Logout
    </button>

</header>
<div id="loginModal" class="login-modal">
  <div class="login-card">
    <h2>Sign in</h2>
    <p style="font-size:0.75rem;color:#9ca3af;margin-bottom:12px;">
      Choose your role and login to access the system.
    </p>

    <label>
      Role
      <select id="roleSelect">
        <option value="doctor">Doctor</option>
        <option value="patient">Patient</option>
        <option value="pharmacy">Pharmacy</option>
      </select>
    </label>

    <label>
      Username
      <input id="usernameInput" placeholder="e.g. doc1" />
    </label>

    <label>
      Password
      <input id="passwordInput" type="password" placeholder="••••••" />
    </label>

    <label id="loginPatientRow">
      Patient ID (for doctor / pharmacy)
      <input id="loginPatientIdInput" placeholder="e.g. P001" />
    </label>

    <button id="btnLogin" class="btn btn-primary" style="width:100%;margin-top:8px;">
      Login
    </button>
  </div>
</div>


  <main>
    <div id="statusLine" class="status-line"></div>
    <div id="doctorView">
    <!-- Stepper / Carousel header -->
    <div id="doctorAccessBadge"
        style="font-size:0.75rem;color:#f87171;margin-left:12px;">
    </div>
    <div class="stepper">
      <div id="step1Pill" class="step-pill active">
        <span class="index">1</span>
        <span>Patient Presence Agent (Face Verify)</span>
      </div>
      <div id="step2Pill" class="step-pill">
        <span class="index">2</span>
        <span>Listening Agents (Audio / Live)</span>
      </div>
      <div id="step3Pill" class="step-pill">
        <span class="index">3</span>
        <span>Clinical Brain & EMR Agents</span>
      </div>

      <div class="step-nav">
        <button id="btnPrev" class="btn btn-ghost" disabled>← Back</button>
        <button id="btnNext" class="btn btn-primary">Next →</button>
      </div>
    </div>

    <!-- Slide 1: Biometric Gate -->
    <section id="slide1" class="slide active">
      <div class="grid" style="align-items:flex-start; margin-bottom:16px;">
        <div class="card">
          <h2>🔐 Patient Presence Agent</h2>
          <p>This agent verifies that the patient is physically present using face detection. Only after this step, listening agents and EMR will unlock.</p>
          <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:6px;">
            <button id="btnStartCam" class="btn btn-ghost">🎥 Start Camera</button>
            <button id="btnVerifyFace" class="btn btn-primary">✅ Verify Patient Face</button>
          </div>
          <p style="margin-top:8px;font-size:0.8rem;">
            Status:
            <span id="verifyStatusText" style="font-weight:600; color:#f97373;">Not Verified</span>
          </p>
          <p style="margin-top:4px;font-size:0.75rem;">
            Once verified, Step 2 (Listening) and Step 3 (Clinical Brain & EMR) become available for this patient.
          </p>
        </div>
        <div class="card">
          <h2>Patient EHR Snapshot</h2>
          <button id="btnRequestAccess" class="btn btn-ghost" style="margin-top:6px;">
            Request EHR Access
          </button>
          <div id="ehrDemoBox">
            <p style="font-size:0.8rem;color:#9ca3af;">
              Verify patient face to load EHR demographics & insurance.
            </p>
          </div>
        </div>
        <div class="card" style="display:flex;flex-direction:column;align-items:flex-start;gap:6px;">
          <h3>Patient Camera View</h3>
          <video id="verifyVideo" autoplay muted></video>
          <p style="font-size:0.75rem;color:#9ca3af;">Align the patient's face in the frame before clicking <b>Verify Patient Face</b>.</p>
        </div>
      </div>
    </section>

    <!-- Slide 2: Listening Agents (Upload + Live Mic) -->
    <section id="slide2" class="slide">
      <div class="grid">
        <div class="card">
          <h2>🎧 Upload Consultation Audio (WAV)</h2>
          <p>This triggers the Audio Scribe Agent. Audio is transcribed, then passed into the workflow graph.</p>
          <input id="audioFileInput" type="file" accept=".wav,audio/wav" style="margin-top:6px;font-size:0.8rem;" />
          <p id="audioFileName" style="font-size:0.75rem;margin-top:4px;color:#e5e7eb;"></p>
          <button id="btnRunAudio" class="btn btn-primary" style="margin-top:10px;">▶ Run Audio Workflow</button>
        </div>
        <div class="card">
          <h2>🎙️ Live Microphone Agent</h2>
          <p>Uses browser speech recognition to capture the consultation in real time.</p>
          <p id="liveSupportMsg" style="font-size:0.75rem;"></p>
          <div style="display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;">
            <button id="btnStartListening" class="btn btn-primary">Start Listening</button>
            <button id="btnStopListening" class="btn btn-danger">Stop</button>
          </div>
          <h3 style="margin-top:10px;">Live Transcript (editable)</h3>
          <textarea id="liveTranscriptBox" placeholder="Speak after clicking Start Listening..."></textarea>
          <button id="btnRunLiveFromText" class="btn btn-ghost" style="margin-top:8px;">🤖 Run Workflow on Transcript</button>
        </div>
        <div class="card">
          <h3>Transcript from Audio (editable)</h3>
          <p>Transcript will appear here after you upload audio and run the workflow. You can edit and re-run on text.</p>
          <textarea id="transcriptBox" placeholder="Transcript will appear here after audio upload or live listening..."></textarea>
          <button id="btnRunFromText" class="btn btn-ghost" style="margin-top:8px;">🤖 Run Workflow on Transcript</button>
        </div>
      </div>
    </section>

    <!-- Slide 3: Clinical Brain & EMR -->
    <section id="slide3" class="slide">
      <div class="grid" style="margin-bottom:16px;">
        <div class="card">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <h2>🗂 EMR Records for this Patient</h2>
            <button id="btnLoadEmr" class="btn btn-primary">Refresh EMR</button>
          </div>
          <div id="emrList" style="margin-top:8px;max-height:260px;overflow-y:auto;"></div>
        </div>
            <div class="card">
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <h3>🏥 Pharmacy Orders</h3>
          <button id="btnLoadPharmacy" class="btn btn-ghost">Refresh</button>
        </div>
        <div id="pharmacyList" style="margin-top:8px;max-height:260px;overflow-y:auto;font-size:0.8rem;"></div>
        </div>

        <div class="card">
          <h3>🩺 Symptoms</h3>
          <div id="symptomList" style="font-size:0.8rem;color:#e5e7eb;">
            <p style="color:#9ca3af;font-size:0.8rem;">No symptoms yet. Run a workflow.</p>
          </div>
        </div>
        <div class="card">
          <h3>🧪Suggested Investigations</h3>
          <div id="testList">
            <p style="color:#9ca3af;font-size:0.8rem;">No tests yet. Mention symptoms like "chest pain", "fever", "diabetes".</p>
          </div>
          <p style="margin-top:6px;font-size:0.75rem;color:#9ca3af;">
            Doctor can edit the final list below (one test per line) before approval:
          </p>
          <textarea id="testsEditBox" placeholder="ECG 12-lead&#10;Chest X-Ray (PA view)&#10;CBC with Differential" style="margin-top:4px;font-size:0.75rem;min-height:90px;"></textarea>
        </div>
        <div class="card">
        <h3>Draft Prescription</h3>
        <div id="rxBox">
          <p style="color:#9ca3af;font-size:0.8rem;">No draft prescription yet.</p>
        </div>

        <p style="margin-top:6px;font-size:0.75rem;color:#9ca3af;">
          Doctor can edit the prescription before saving to EMR:
        </p>
        <textarea id="rxEditBox" placeholder="Edited prescription will appear here after workflow..." style="margin-top:4px;font-size:0.75rem;min-height:100px;"></textarea>

        <div id="safetyBox" style="margin-top:6px;"></div>
        <div id="emrIdBox" style="margin-top:6px;font-size:0.75rem;"></div>

        <button id="btnApproveEmr" class="btn btn-primary" style="margin-top:8px;width:100%;">
          Approve & Save to EMR
        </button>
        <button id="btnSendPharmacy" class="btn btn-ghost" style="margin-top:6px;width:100%;">
          Send Prescription to Pharmacy
        </button>
      </div>

        <div class="card" style="grid-column:1/-1;">
          <h3>Workflow Timeline (Audit Log)</h3>
          <div id="auditLogBox" style="max-height:160px;overflow-y:auto;font-size:0.75rem;color:#e5e7eb;">
            <p style="color:#9ca3af;font-size:0.8rem;">Run a workflow to view events here.</p>
          </div>
        </div>
      </div>
    </section>
    </div> 
  <div id="patientView" style="display:none;">
    <div class="card">
      <h2>Patient EMR Portal</h2>
      <div class="card" style="margin-top:16px;">
        <h3>Doctor Access Control</h3>
        <p style="font-size:0.8rem;color:#9ca3af;">
          Choose which doctors can view your EHR.
        </p>

        <button id="btnLoadAccessList" class="btn btn-ghost" style="margin-bottom:8px;">
          Refresh Doctor Access List
        </button>

        <div id="doctorAccessList" style="font-size:0.8rem;"></div>
      </div>
      <button id="btnGrantAccess" class="btn btn-primary" style="margin-top:10px;">
        Grant Doctor Access
      </button>
      <p style="font-size:0.8rem;color:#9ca3af;">
        You are logged in as a patient. You can view your EMR records, but cannot modify them.
      </p>
      <button id="btnPatientLoadEmr" class="btn btn-primary" style="margin-top:8px;">Refresh My EMR</button>
      <div id="patientEmrList" style="margin-top:10px;max-height:320px;overflow-y:auto;font-size:0.8rem;"></div>
    </div>
  </div>

  <!-- PHARMACY VIEW: pharmacy orders only -->
  <div id="pharmacyView" style="display:none;">
    <div class="card">
      <h2>Pharmacy Orders Console</h2>
      <p style="font-size:0.8rem;color:#9ca3af;">
        You are logged in as pharmacy. You can see doctor-approved prescriptions sent for dispensing.
      </p>
      <div style="display:flex;gap:8px;align-items:center;margin-top:8px;">
        <span style="font-size:0.75rem;color:#9ca3af;">Filter by Patient ID (optional):</span>
        <input id="pharmacyPatientFilter" placeholder="P001"
               style="padding:4px 8px;border-radius:999px;border:none;background:#020617;color:#e5e7eb;font-size:0.75rem;">
        <button id="btnPharmacyRefresh" class="btn btn-primary" style="font-size:0.75rem;padding:4px 10px;">
          Refresh Orders
        </button>
      </div>
      <div id="pharmacyOrdersList" style="margin-top:10px;max-height:320px;overflow-y:auto;font-size:0.8rem;"></div>
    </div>
  </div>

  </main>
  <script>
    // ---------- Global Elements ----------
    const statusLine = document.getElementById("statusLine");
    const patientIdInput = document.getElementById("patientIdInput");

    const step1Pill = document.getElementById("step1Pill");
    const step2Pill = document.getElementById("step2Pill");
    const step3Pill = document.getElementById("step3Pill");
    const btnPrev = document.getElementById("btnPrev");
    const btnNext = document.getElementById("btnNext");

    const slide1 = document.getElementById("slide1");
    const slide2 = document.getElementById("slide2");
    const slide3 = document.getElementById("slide3");

    const btnStartCam = document.getElementById("btnStartCam");
    const btnVerifyFace = document.getElementById("btnVerifyFace");
    const verifyStatusText = document.getElementById("verifyStatusText");
    const videoEl = document.getElementById("verifyVideo");

    const audioFileInput = document.getElementById("audioFileInput");
    const audioFileName = document.getElementById("audioFileName");
    const btnRunAudio = document.getElementById("btnRunAudio");
    const transcriptBox = document.getElementById("transcriptBox");
    const btnRunFromText = document.getElementById("btnRunFromText");

    const liveSupportMsg = document.getElementById("liveSupportMsg");
    const btnStartListening = document.getElementById("btnStartListening");
    const btnStopListening = document.getElementById("btnStopListening");
    const liveTranscriptBox = document.getElementById("liveTranscriptBox");
    const btnRunLiveFromText = document.getElementById("btnRunLiveFromText");

    const btnLoadEmr = document.getElementById("btnLoadEmr");
    const emrList = document.getElementById("emrList");
    const btnLoadPharmacy = document.getElementById("btnLoadPharmacy");
    const pharmacyList = document.getElementById("pharmacyList");
    const symptomList = document.getElementById("symptomList");
    const testList = document.getElementById("testList");
    const testsEditBox = document.getElementById("testsEditBox");
    const rxBox = document.getElementById("rxBox");
    const rxEditBox = document.getElementById("rxEditBox");
    const safetyBox = document.getElementById("safetyBox");
    const emrIdBox = document.getElementById("emrIdBox");
    const auditLogBox = document.getElementById("auditLogBox");
    const btnApproveEmr = document.getElementById("btnApproveEmr");
    const btnSendPharmacy = document.getElementById("btnSendPharmacy");

    // Role / login elements
    const roleSelect = document.getElementById("roleSelect");
    const usernameInput = document.getElementById("usernameInput");
    const passwordInput = document.getElementById("passwordInput");
    const btnLogin = document.getElementById("btnLogin");
    const btnLogout = document.getElementById("btnLogout");
    const currentUserInfo = document.getElementById("currentUserInfo");
    const loginModal = document.getElementById("loginModal");
    const loginPatientRow = document.getElementById("loginPatientRow");
    const loginPatientIdInput = document.getElementById("loginPatientIdInput");
    // Patient & pharmacy portal elements
    const btnPatientLoadEmr = document.getElementById("btnPatientLoadEmr");
    const patientEmrList = document.getElementById("patientEmrList");
    const btnPharmacyRefresh = document.getElementById("btnPharmacyRefresh");
    const pharmacyPatientFilter = document.getElementById("pharmacyPatientFilter");
    const pharmacyOrdersList = document.getElementById("pharmacyOrdersList");
    const ehrDemoBox = document.getElementById("ehrDemoBox");
    const btnGrantAccess = document.getElementById("btnGrantAccess");
    const btnLoadAccessList = document.getElementById("btnLoadAccessList");
    const doctorAccessList = document.getElementById("doctorAccessList");
    const btnRequestAccess = document.getElementById("btnRequestAccess");
    // ---------- Demo users ----------
    const USERS = [
      { username: "doc1",   password: "doc123",    role: "doctor"   },
      { username: "doc2",   password: "doc123",    role: "doctor"   },
      { username: "doc3",   password: "doc123",    role: "doctor"   },
      { username: "pat1",   password: "pat123",    role: "patient",  patient_id: "P001" },
      { username: "pat2",   password: "pat123",    role: "patient",  patient_id: "P002" },
      { username: "pat3",   password: "pat123",    role: "patient",  patient_id: "P003" },
      { username: "pat4",   password: "pat123",    role: "patient",  patient_id: "P004" },
      { username: "pat5",   password: "pat123",    role: "patient",  patient_id: "P005" },
      { username: "pat6",   password: "pat123",    role: "patient",  patient_id: "P006" },
      { username: "pat7",   password: "pat123",    role: "patient",  patient_id: "P007" },
      { username: "pat8",   password: "pat123",    role: "patient",  patient_id: "P008" },
      { username: "pat9",   password: "pat123",    role: "patient",  patient_id: "P009" },
      { username: "pat10",  password: "pat123",    role: "patient",  patient_id: "P010" },
      { username: "pharma1",password: "pharma123", role: "pharmacy" },
    ];

    // ---------- Global state ----------
    let currentUser = null;         // { username, role, ... }
    let currentRole = null;         // "doctor" | "patient" | "pharmacy"
    let patientVerified = false;
    let currentState = null;        // latest workflow state from backend
    let currentStream = null;       // camera stream
    let currentSlide = 1;
    let lastApprovedEmrId = null;

    let recognition = null;         // browser STT object
    let listening = false;
    let finalTranscript = "";

    // ---------- Helpers ----------
    if (btnRequestAccess) {
    btnRequestAccess.onclick = async () => {
        if (!currentUser || currentRole !== "doctor") return;

        const pid = getPatientId();
        const uname = currentUser.username;

        const res = await fetch(
            `/doctor/request-access?patient_id=${pid}&doctor_username=${uname}`,
            { method: "POST" }
        );
        const json = await res.json();
        setStatus(json.message, "info");
        };
    }
    if (btnLoadAccessList) {
        btnLoadAccessList.onclick = async () => {
            const pid = currentUser.patient_id;
            const res = await fetch(`/patient/access-list?patient_id=${pid}`);
            const json = await res.json();
            doctorAccessList.innerHTML = "";
            json.access_list.forEach(item => {
                const div = document.createElement("div");
                div.style.marginBottom = "6px";

                div.innerHTML = `
                    <b>${item.doctor_username}</b>
                    <span style="color:${item.is_allowed ? "#4ade80" : "#f87171"};">
                        (${item.is_allowed ? "Access Granted" : "Access Blocked"})
                    </span>
                    <button class="btn btn-primary" style="padding:2px 8px;margin-left:8px;"
                            onclick="toggleAccess('${pid}', '${item.doctor_username}', ${item.is_allowed})">
                      ${item.is_allowed ? "Revoke" : "Grant"}
                    </button>
                `;
                doctorAccessList.appendChild(div);
            });
        };
    }
    if (btnGrantAccess) {
        btnGrantAccess.onclick = async () => {
            if (!currentUser || currentRole !== "patient") {
                setStatus("Only patients can grant access.", "warn");
                return;
            }
            const pid = currentUser.patient_id;
            const doctorUsername = prompt("Enter Doctor Username to grant access:");
            if (!doctorUsername) {
                setStatus("Doctor username required.", "warn");
                return;
            }
            try {
                const res = await fetch(
                    `/patient/grant-access?patient_id=${pid}&doctor_username=${doctorUsername}`,
                    { method: "POST" }
                );
                const json = await res.json();
                setStatus(json.message, "ok");
            } catch (err) {
                console.error(err);
                setStatus("Error granting access: " + err.message, "warn");
            }
        };
    }
    roleSelect.onchange = () => {
      const role = roleSelect.value;
      if (role === "patient") {
        loginPatientRow.style.display = "none";
      } else {
        loginPatientRow.style.display = "block";
      }
    };
    roleSelect.onchange(); // set initial state
    
    async function loadEhrForPatient(pid) {
      console.log("LOADING EHR FOR:", pid, "ROLE:", currentRole, "USER:", currentUser);

      ehrDemoBox.innerHTML = "Loading EHR...";

      if (!currentUser || !currentRole) {
        ehrDemoBox.innerHTML = "Not logged in";
        return;
      }

      const params = new URLSearchParams();
      params.set("role", currentRole);
      params.set("username", currentUser.username);

      const url = `/ehr/${encodeURIComponent(pid)}?${params.toString()}`;

      try {
        const res = await fetch(url);
        if (!res.ok) {
          const err = await res.text();
          throw new Error("Backend error " + res.status + ": " + err);
        }
        const data = await res.json();
        if (data.exists) {
          if (window.currentRole === "doctor") {
              document.getElementById("doctorAccessBadge").textContent =
                "✔ Access Granted";
              document.getElementById("doctorAccessBadge").style.color = "#4ade80";
          }
        }
        renderEhrSummary(data);
      } catch (err) {
      document.getElementById("doctorAccessBadge").textContent ="❌ Access Denied";
      document.getElementById("doctorAccessBadge").style.color = "#f87171";
      ehrDemoBox.innerHTML = "Error loading EHR: " + err.message;
      }
    }

    async function toggleAccess(pid, doctorUsername, currentlyAllowed) {
      const endpoint = currentlyAllowed
        ? `/patient/revoke-access?patient_id=${pid}&doctor_username=${doctorUsername}`
        : `/patient/grant-access?patient_id=${pid}&doctor_username=${doctorUsername}`;
        const res = await fetch(endpoint, { method: "POST" });
        const json = await res.json();
        setStatus(json.message, "ok");
        btnLoadAccessList.click();
    }

    function renderEhrSummary(ehr) {
      if (!ehrDemoBox) return;

      if (!ehr || !ehr.exists) {
        ehrDemoBox.innerHTML =
          "<p style='font-size:0.8rem;color:#9ca3af;'>No EHR found for this patient.</p>";
        return;
      }

      var d = ehr.demographics || {};
      var ins = ehr.insurance || null;

      var name = d.full_name || ehr.patient_id || "";
      var dob = d.date_of_birth || "—";
      var gender = d.gender || "—";
      var phone = d.phone || "—";
      var email = d.email || "";
      var emerg =
        (d.emergency_contact_name || "—") +
        " (" +
        (d.emergency_contact_phone || "—") +
        ")";
      var blood = d.blood_group || "—";

      var allergies = d.allergies || [];
      var chronic = d.chronic_conditions || [];
      var html = "";

      html += "<p class='ehr-label'>Name</p>";
      html += "<p>" + name + "</p>";

      html += "<p class='ehr-label'>DOB / Gender</p>";
      html += "<p>" + dob + " · " + gender + "</p>";

      html += "<p class='ehr-label'>Contact</p>";
      html += "<p>" + phone + (email ? " · " + email : "") + "</p>";

      html += "<p class='ehr-label'>Emergency Contact</p>";
      html += "<p>" + emerg + "</p>";
      if (allergies.length > 0) {
        html += "<p class='ehr-label'>Allergies</p>";
        html += "<p>" + allergies.join(", ") + "</p>";
      }

      if (chronic.length > 0) {
        html += "<p class='ehr-label'>Chronic Conditions</p>";
        html += "<p>" + chronic.join(", ") + "</p>";
      }

      if (ins) {
        html +=
          "<hr style='border-color:#1f2937;margin:6px 0;' />";
        html += "<p class='ehr-label'>Insurance</p>";
        html +=
          "<p>" +
          (ins.provider_name || "") +
          " · " +
          (ins.policy_number || "") +
          "</p>";

        if (ins.coverage_details) {
          html +=
            "<p style='font-size:0.75rem;color:#9ca3af;'>" +
            ins.coverage_details +
            "</p>";
        }
        if (ins.billing_notes) {
          html +=
            "<p style='font-size:0.7rem;color:#6b7280;margin-top:2px;'>" +
            ins.billing_notes +
            "</p>";
        }
      }

      ehrDemoBox.innerHTML = html;
    }

    
    function setStatus(text, type = "info") {
      statusLine.textContent = text || "";
      statusLine.className = "status-line";
      if (!text) return;
      if (type === "ok")   statusLine.classList.add("status-ok");
      if (type === "warn") statusLine.classList.add("status-warn");
      if (type === "info") statusLine.classList.add("status-info");
    }

    function getPatientId() {
      return (patientIdInput.value || "").trim();
    }

    function getOrderId(rec) {
      return rec.order_id || rec.pharmacy_order_id || "";
    }

    function renderCurrentUserInfo() {
      if (!currentUser || !currentRole) {
        currentUserInfo.style.display = "none";
        currentUserInfo.textContent = "Not logged in";
        return;
      }

      currentUserInfo.style.display = "inline-flex";

      let roleLabel = "";
      if (currentRole === "doctor")   roleLabel = "Doctor";
      if (currentRole === "patient")  roleLabel = "Patient";
      if (currentRole === "pharmacy") roleLabel = "Pharmacy";

      let extra = "";
      if (currentRole === "patient" && currentUser.patient_id) {
        extra = " · Patient ID: " + currentUser.patient_id;
      }

      currentUserInfo.textContent = roleLabel + ": " + currentUser.username + extra;
    }

    function updateRoleUI() {
      const doctorView = document.getElementById("doctorView");
      const patientView = document.getElementById("patientView");
      const pharmacyView = document.getElementById("pharmacyView");

      doctorView.style.display = "none";
      patientView.style.display = "none";
      pharmacyView.style.display = "none";

      // Show modal if not logged in
      if (!currentRole) {
        loginModal.style.display = "flex";
        currentUserInfo.style.display = "none";
        renderCurrentUserInfo();
        return;
      }

      // Logged in → hide modal
      loginModal.style.display = "none";

      if (currentRole === "doctor") {
        doctorView.style.display = "";
        patientIdInput.disabled = false;
        setStatus("Logged in as DOCTOR. Face verification required to access EMR & pharmacy.", "info");
      } else if (currentRole === "patient") {
        patientView.style.display = "";
        patientIdInput.disabled = true;
        setStatus("Logged in as PATIENT. You can view your EMR anytime.", "info");
      } else if (currentRole === "pharmacy") {
        pharmacyView.style.display = "";
        patientIdInput.disabled = false;
        setStatus("Logged in as PHARMACY. You can only view pharmacy orders.", "info");
      }

      renderCurrentUserInfo();
      currentUserInfo.style.display = "inline-flex";
    }


    function updateStepUI() {
      [step1Pill, step2Pill, step3Pill].forEach(p => p.classList.remove("active"));
      [slide1, slide2, slide3].forEach(s => s.classList.remove("active"));

      if (currentSlide === 1) {
        step1Pill.classList.add("active");
        slide1.classList.add("active");
        btnPrev.disabled = true;
        btnNext.textContent = "Next →";
      } else if (currentSlide === 2) {
        step2Pill.classList.add("active");
        slide2.classList.add("active");
        btnPrev.disabled = false;
        btnNext.textContent = "Next →";
      } else {
        step3Pill.classList.add("active");
        slide3.classList.add("active");
        btnPrev.disabled = false;
        btnNext.textContent = "Done";
      }
    }

    function goNext() {
      if (currentSlide === 1) {
        if (!patientVerified) {
          setStatus("Patient presence not verified. Please verify face before moving to Listening Agents.", "warn");
          return;
        }
        currentSlide = 2;
      } else if (currentSlide === 2) {
        currentSlide = 3;
      } else {
        currentSlide = 1;
      }
      updateStepUI();
    }

    function goPrev() {
      if (currentSlide === 2) currentSlide = 1;
      else if (currentSlide === 3) currentSlide = 2;
      updateStepUI();
    }

    btnNext.onclick = goNext;
    btnPrev.onclick = goPrev;

    // When patient changes, reset verification + slide
    patientIdInput.addEventListener("input", () => {
      patientVerified = false;
      verifyStatusText.textContent = "Not Verified";
      verifyStatusText.style.color = "#f97373";
      currentSlide = 1;
      lastApprovedEmrId = null;
      updateStepUI();
      setStatus("Patient changed. Please verify face again to unlock next steps.", "info");
    });

    // ---------- Biometric gate ----------
    btnStartCam.onclick = async () => {
      try {
        if (currentStream) {
          currentStream.getTracks().forEach(t => t.stop());
          currentStream = null;
        }
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        currentStream = stream;
        videoEl.srcObject = stream;
        await videoEl.play();
        setStatus("Camera started. Align patient's face and click Verify.", "info");
      } catch (err) {
        console.error(err);
        setStatus("Cannot access camera: " + err.message, "warn");
      }
    };

    btnVerifyFace.onclick = async () => {
      if (currentRole !== "doctor") {
        setStatus("Only doctors can perform face verification.", "warn");
        return;
      }

      const pid = getPatientId();
      if (!pid) {
        setStatus("Please enter a Patient ID before verifying.", "warn");
        return;
      }
      if (!videoEl.videoWidth) {
        setStatus("Camera not ready. Click Start Camera first.", "warn");
        return;
      }

      const canvas = document.createElement("canvas");
      canvas.width = videoEl.videoWidth || 640;
      canvas.height = videoEl.videoHeight || 480;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);

      setStatus("Verifying patient face...", "info");

      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg"));
      const formData = new FormData();
      formData.append("image", blob, "frame.jpg");

      try {
        const res = await fetch("/verify-patient-face?patient_id=" + encodeURIComponent(pid), {
          method: "POST",
          body: formData
        });
        const json = await res.json();
        if (json.authorized) {
          patientVerified = true;
          verifyStatusText.textContent = "Verified";
          verifyStatusText.style.color = "#4ade80";
          setStatus("Patient face verified. You can move to Listening Agents.", "ok");
          if (ehrDemoBox) {
            loadEhrForPatient(pid);
          }
        } else {
          patientVerified = false;
          verifyStatusText.textContent = "Not Verified";
          verifyStatusText.style.color = "#f97373";
          setStatus("Face verification failed: " + (json.reason || "unknown reason"), "warn");
        }
      } catch (err) {
        console.error(err);
        setStatus("Error during face verification: " + err.message, "warn");
      } finally {
        if (currentStream) {
          currentStream.getTracks().forEach(t => t.stop());
          currentStream = null;
        }
      }
    };

    // ---------- Audio upload workflow ----------
    audioFileInput.onchange = () => {
      const file = audioFileInput.files[0];
      audioFileName.textContent = file ? ("Selected: " + file.name) : "";
    };

    btnRunAudio.onclick = async () => {
      if (currentRole !== "doctor") {
        setStatus("Only doctors can perform this action.", "warn");
        return;
      }
      if (!patientVerified) {
        setStatus("EMR/workflow locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      const file = audioFileInput.files[0];
      if (!file) {
        setStatus("Select a WAV file first.", "warn");
        return;
      }

      setStatus("Uploading audio and running workflow...", "info");
      const formData = new FormData();
      formData.append("audio", file);

      try {
        const res = await fetch("/audio-workflow?patient_id=" + encodeURIComponent(pid), {
          method: "POST",
          body: formData
        });
        if (!res.ok) {
          const text = await res.text();
          throw new Error("Backend error " + res.status + ": " + text);
        }
        const json = await res.json();
        currentState = json.state;
        transcriptBox.value = currentState.raw_transcript || "";
        liveTranscriptBox.value = currentState.raw_transcript || "";
        renderState();
        setStatus("Audio workflow completed.", "ok");
        currentSlide = 3;
        updateStepUI();
      } catch (err) {
        console.error(err);
        setStatus("Error calling /audio-workflow: " + err.message, "warn");
      }
    };

    // ---------- Workflow from transcript ----------
    async function runWorkflowWithTranscript(text) {
      if (currentRole !== "doctor") {
        setStatus("Only doctors can perform this action.", "warn");
        return;
      }
      if (!patientVerified) {
        setStatus("EMR/workflow locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      text = (text || "").trim();
      if (!text) {
        setStatus("Transcript is empty. Speak / type something first.", "warn");
        return;
      }

      setStatus("Running workflow on transcript...", "info");
      try {
        const res = await fetch("/trigger-workflow", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ patient_id: pid, note_text: text })
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        currentState = json.state;
        renderState();
        setStatus("Workflow completed.", "ok");
        currentSlide = 3;
        updateStepUI();
      } catch (err) {
        console.error(err);
        setStatus("Error calling /trigger-workflow: " + err.message, "warn");
      }
    }

    btnRunFromText.onclick = () => runWorkflowWithTranscript(transcriptBox.value);
    btnRunLiveFromText.onclick = () => runWorkflowWithTranscript(liveTranscriptBox.value);

    // ---------- Browser STT (live mic) ----------
    (function initSTT() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        liveSupportMsg.textContent = "❌ This browser does not support SpeechRecognition. Use Chrome.";
        btnStartListening.disabled = true;
        btnStopListening.disabled = true;
        return;
      }
      recognition = new SR();
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";
      liveSupportMsg.textContent = "✅ Live STT available (Chrome Web Speech).";

      recognition.onstart = () => {
        listening = true;
        setStatus("Listening... speak now.", "info");
      };
      recognition.onerror = (event) => {
        console.error("STT error:", event);
        setStatus("Speech recognition error: " + event.error, "warn");
      };
      recognition.onend = () => {
        listening = false;
        setStatus("Stopped listening.", "info");
      };
      recognition.onresult = (event) => {
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; ++i) {
          const t = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            finalTranscript += " " + t;
          } else {
            interim += " " + t;
          }
        }
        liveTranscriptBox.value = (finalTranscript + " " + interim).trim();
        transcriptBox.value = liveTranscriptBox.value;
      };
    })();

    btnStartListening.onclick = () => {
      if (!recognition || listening) return;
      finalTranscript = liveTranscriptBox.value || "";
      recognition.start();
    };

    btnStopListening.onclick = () => {
      if (!recognition || !listening) return;
      recognition.stop();
    };

    // ---------- Doctor: EMR and Pharmacy (per-patient) ----------
    btnLoadEmr.onclick = async () => {
      if (!patientVerified) {
        setStatus("EMR locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      setStatus("Loading EMR records...", "info");
      try {
        const res = await fetch("/get-emr?patient_id=" + encodeURIComponent(pid));
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        renderEmrList(json);
        setStatus("Loaded " + json.length + " EMR record(s).", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Error loading EMR: " + err.message, "warn");
      }
    };

    btnLoadPharmacy.onclick = async () => {
      if (!patientVerified) {
        setStatus("Pharmacy data locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      setStatus("Loading pharmacy orders...", "info");
      try {
        const res = await fetch("/get-pharmacy-orders?patient_id=" + encodeURIComponent(pid));
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        renderPharmacyList(json);
        setStatus("Loaded " + json.length + " pharmacy order(s).", "ok");
      } catch (err) {
        console.error(err);
        setStatus("Error loading pharmacy orders: " + err.message, "warn");
      }
    };

    // ---------- Patient portal ----------
    if (btnPatientLoadEmr) {
      btnPatientLoadEmr.onclick = async () => {
        if (!currentRole || currentRole !== "patient" || !currentUser) {
          setStatus("Login as patient to view EMR.", "warn");
          return;
        }
        const pid = currentUser.patient_id;
        if (!pid) {
          setStatus("No patient_id bound to this account.", "warn");
          return;
        }
        setStatus("Loading your EMR records...", "info");
        try {
          const res = await fetch("/get-emr?patient_id=" + encodeURIComponent(pid));
          if (!res.ok) {
            const t = await res.text();
            throw new Error("Backend error " + res.status + ": " + t);
          }
          const json = await res.json();
          renderPatientEmrList(json);
          setStatus("Loaded " + json.length + " EMR records.", "ok");
        } catch (err) {
          console.error(err);
          setStatus("Error loading EMR: " + err.message, "warn");
        }
      };
    }

    // ---------- Pharmacy console ----------
    if (btnPharmacyRefresh) {
      btnPharmacyRefresh.onclick = async () => {
        if (!currentRole || currentRole !== "pharmacy") {
          setStatus("Login as pharmacy to view orders.", "warn");
          return;
        }

        const pid = (pharmacyPatientFilter.value || "").trim();
        setStatus("Loading pharmacy orders...", "info");

        try {
          let url = "/get-pharmacy-orders";
          if (pid) {
            url += "?patient_id=" + encodeURIComponent(pid);
          }
          const res = await fetch(url);
          if (!res.ok) {
            const t = await res.text();
            throw new Error("Backend error " + res.status + ": " + t);
          }
          const json = await res.json();
          renderPharmacyOrdersList(json);
          setStatus("Loaded " + json.length + " pharmacy order(s).", "ok");
        } catch (err) {
          console.error(err);
          setStatus("Error loading pharmacy orders: " + err.message, "warn");
        }
      };
    }

    // ---------- Render helpers ----------
    function renderPatientEmrList(records) {
      patientEmrList.innerHTML = "";
      if (!records || records.length === 0) {
        patientEmrList.innerHTML =
          "<p style='font-size:0.8rem;color:#9ca3af;'>No EMR records found for your account.</p>";
        return;
      }
      records.forEach(rec => {
        const div = document.createElement("div");
        div.className = "emr-item";
        const ts = (rec.timestamp_utc || "").replace("T", " ").replace("Z", "");
        const note = rec.note_summary || "";
        const shortNote = note.length > 160 ? note.slice(0, 160) + "..." : note;
        div.innerHTML =
          "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>" +
          "<span style='font-family:monospace;color:#38bdf8;font-size:0.8rem;'>" + (rec.emr_record_id || "") + "</span>" +
          "<span style='font-size:0.7rem;color:#9ca3af;'>" + ts + "</span>" +
          "</div>" +
          "<p style='font-size:0.75rem;color:#e5e7eb;'>" + shortNote + "</p>";
        patientEmrList.appendChild(div);
      });
    }

    function renderPharmacyOrdersList(records) {
      pharmacyOrdersList.innerHTML = "";
      if (!records || records.length === 0) {
        pharmacyOrdersList.innerHTML =
          "<p style='font-size:0.8rem;color:#9ca3af;'>No pharmacy orders found.</p>";
        return;
      }
      records.forEach(rec => {
        const div = document.createElement("div");
        div.className = "emr-item";
        const ts = (rec.timestamp_utc || "").replace("T", " ").replace("Z", "");
        const status = rec.status || "pending";
        const pid = rec.patient_id || "";
        let rxPreview = "";
        const rx = rec.prescription || rec.draft_prescription || "";
        if (rx) {
          const lines = rx.split("\\n");
          rxPreview = lines.slice(0, 4).join("\\n");
          if (lines.length > 4) rxPreview += "\\n...";
        }

        div.innerHTML =
          "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>" +
          "<span style='font-family:monospace;color:#fbbf24;font-size:0.8rem;'>" + getOrderId(rec) + "</span>" +
          "<span style='font-size:0.7rem;color:#9ca3af;'>" + ts + "</span>" +
          "</div>" +
          "<p style='font-size:0.75rem;color:#e5e7eb;'><b>Patient:</b> " + pid + "</p>" +
          "<p style='font-size:0.75rem;color:#e5e7eb;'><b>Status:</b> " + status + "</p>" +
          (rec.emr_record_id
            ? "<p style='font-size:0.75rem;color:#9ca3af;'><b>EMR:</b> " + rec.emr_record_id + "</p>"
            : "");

        if (rxPreview) {
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "Prescription details";
          summary.style.cursor = "pointer";
          summary.style.fontSize = "0.75rem";
          summary.style.color = "#38bdf8";
          const pre = document.createElement("pre");
          pre.textContent = rxPreview;
          pre.style.marginTop = "4px";
          details.appendChild(summary);
          details.appendChild(pre);
          div.appendChild(details);
        }
        pharmacyOrdersList.appendChild(div);
      });
    }

    function renderEmrList(records) {
      emrList.innerHTML = "";
      if (!records || records.length === 0) {
        emrList.innerHTML =
          "<p style='font-size:0.8rem;color:#9ca3af;'>No EMR records yet for this patient.</p>";
        return;
      }
      records.slice().reverse().forEach(rec => {
        const div = document.createElement("div");
        div.className = "emr-item";
        const ts = (rec.timestamp_utc || "").replace("T", " ").replace("Z", "");
        const sym = rec.symptoms && rec.symptoms.length ? rec.symptoms.join(", ") : "None";
        const tests = rec.suggested_tests && rec.suggested_tests.length ? rec.suggested_tests.join(", ") : "None";
        let inner =
          "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>" +
          "<span style='font-family:monospace;color:#38bdf8;font-size:0.8rem;'>" + (rec.emr_record_id || "") + "</span>" +
          "<span style='font-size:0.7rem;color:#9ca3af;'>" + ts + "</span>" +
          "</div>" +
          "<p style='font-size:0.8rem;color:#e5e7eb;'><b>Symptoms:</b> " + sym + "</p>" +
          "<p style='font-size:0.8rem;color:#e5e7eb;'><b>Tests:</b> " + tests + "</p>";
        div.innerHTML = inner;

        if (rec.draft_prescription) {
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = rec.approved_by_doctor ? "Approved Prescription" : "Draft Prescription";
          summary.style.cursor = "pointer";
          summary.style.color = rec.approved_by_doctor ? "#4ade80" : "#38bdf8";
          summary.style.fontSize = "0.75rem";
          const pre = document.createElement("pre");
          pre.textContent = rec.draft_prescription;
          pre.style.marginTop = "4px";
          details.appendChild(summary);
          details.appendChild(pre);
          div.appendChild(details);
        }
        emrList.appendChild(div);
      });
    }

    function renderPharmacyList(records) {
      pharmacyList.innerHTML = "";
      if (!records || records.length === 0) {
        pharmacyList.innerHTML =
          "<p style='font-size:0.8rem;color:#9ca3af;'>No pharmacy orders yet for this patient.</p>";
        return;
      }
      records.forEach(rec => {
        const div = document.createElement("div");
        div.className = "emr-item";
        const ts = (rec.timestamp_utc || "").replace("T", " ").replace("Z", "");
        const status = rec.status || "pending";
        let rxPreview = "";
        const rx = rec.prescription || rec.draft_prescription || "";
        if (rx) {
          const lines = rx.split("\\n");
          rxPreview = lines.slice(0, 3).join("\\n");
          if (lines.length > 3) rxPreview += "\\n...";
        }

        div.innerHTML =
          "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>" +
          "<span style='font-family:monospace;color:#fbbf24;font-size:0.8rem;'>" + getOrderId(rec) + "</span>" +
          "<span style='font-size:0.7rem;color:#9ca3af;'>" + ts + "</span>" +
          "</div>" +
          "<p style='font-size:0.75rem;color:#e5e7eb;'><b>Status:</b> " + status + "</p>" +
          (rec.emr_record_id
            ? "<p style='font-size:0.75rem;color:#9ca3af;'><b>From EMR:</b> " + rec.emr_record_id + "</p>"
            : "");

        if (rxPreview) {
          const details = document.createElement("details");
          const summary = document.createElement("summary");
          summary.textContent = "Prescription details";
          summary.style.cursor = "pointer";
          summary.style.fontSize = "0.75rem";
          summary.style.color = "#38bdf8";
          const pre = document.createElement("pre");
          pre.textContent = rxPreview;
          pre.style.marginTop = "4px";
          details.appendChild(summary);
          details.appendChild(pre);
          div.appendChild(details);
        }
        pharmacyList.appendChild(div);
      });
    }

    function renderState() {
      if (!currentState) {
        symptomList.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No symptoms yet. Run a workflow.</p>";
        testList.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No tests yet.</p>";
        rxBox.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No draft prescription yet.</p>";
        safetyBox.innerHTML = "";
        emrIdBox.innerHTML = "";
        auditLogBox.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>Run a workflow to view events here.</p>";
        testsEditBox.value = "";
        rxEditBox.value = "";
        return;
      }

      const s = currentState;

      // Symptoms
      if (Array.isArray(s.symptoms) && s.symptoms.length > 0) {
        symptomList.innerHTML = s.symptoms
          .map(sym => "<span class='badge'>" + sym + "</span>")
          .join(" ");
      } else {
        symptomList.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No symptoms detected.</p>";
      }

      // Tests
      if (Array.isArray(s.suggested_tests) && s.suggested_tests.length > 0) {
        const ul = document.createElement("ul");
        s.suggested_tests.forEach(t => {
          const li = document.createElement("li");
          li.textContent = t;
          ul.appendChild(li);
        });
        testList.innerHTML = "";
        testList.appendChild(ul);
        testsEditBox.value = s.suggested_tests.join("\\n");
      } else {
        testList.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No tests suggested.</p>";
        testsEditBox.value = "";
      }

      // Prescription
      if (s.draft_prescription) {
        rxBox.innerHTML = "<pre>" + s.draft_prescription + "</pre>";
        rxEditBox.value = s.draft_prescription;
      } else {
        rxBox.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No draft prescription yet.</p>";
        rxEditBox.value = "";
      }

      // Safety
      if (Array.isArray(s.safety_flags) && s.safety_flags.length > 0) {
        safetyBox.innerHTML =
          "<p style='font-size:0.75rem;color:#facc15;'><b>⚠ Safety Flags:</b></p>" +
          "<ul>" +
          s.safety_flags.map(f => "<li>" + f + "</li>").join("") +
          "</ul>";
      } else {
        safetyBox.innerHTML = "";
      }

      // EMR ID from executed_actions
      let emrId = null;
      if (Array.isArray(s.executed_actions)) {
        const emrAction = s.executed_actions.find(a => a && a.action === "update_emr");
        if (emrAction && emrAction.emr_record_id) emrId = emrAction.emr_record_id;
      }
      if (emrId) {
        emrIdBox.innerHTML =
          "<span style='font-size:0.75rem;color:#4ade80;'>🗂 EMR stored as <code>" +
          emrId +
          "</code></span>";
      } else {
        emrIdBox.innerHTML = "";
      }

      // Audit log
      if (Array.isArray(s.audit_log) && s.audit_log.length > 0) {
        auditLogBox.innerHTML = "";
        const ol = document.createElement("ol");
        s.audit_log.forEach(line => {
          const li = document.createElement("li");
          li.textContent = line;
          li.style.marginBottom = "2px";
          ol.appendChild(li);
        });
        auditLogBox.appendChild(ol);
      } else {
        auditLogBox.innerHTML =
          "<p style='color:#9ca3af;font-size:0.8rem;'>No audit log entries.</p>";
      }
    }

    // ---------- Send to pharmacy (doctor) ----------
    btnSendPharmacy.onclick = async () => {
      if (currentRole !== "doctor") {
        setStatus("Only doctors can perform this action.", "warn");
        return;
      }
      if (!patientVerified) {
        setStatus("Pharmacy action locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      if (!currentState) {
        setStatus("Run and approve a workflow first.", "warn");
        return;
      }
      const rxText = (rxEditBox.value || "").trim();
      if (!rxText) {
        setStatus("Prescription text is empty. Please review/edit before sending.", "warn");
        return;
      }
      if (!lastApprovedEmrId) {
        setStatus("No approved EMR found. Please approve the consultation before sending to pharmacy.", "warn");
        return;
      }

      const symptoms = Array.isArray(currentState.symptoms) ? currentState.symptoms : [];
      const testsLines = (testsEditBox.value || "")
        .split("\\n")
        .map(t => t.trim())
        .filter(t => t.length > 0);

      setStatus("Sending prescription to pharmacy...", "info");
      try {
        const res = await fetch("/send-to-pharmacy", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: pid,
            prescription: rxText,
            emr_record_id: lastApprovedEmrId,
            suggested_tests: testsLines,
            symptoms: symptoms
          })
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        const orderId = json.order_id || "";
        setStatus("📤 Prescription sent to pharmacy as order " + orderId, "ok");
      } catch (err) {
        console.error(err);
        setStatus("Error sending to pharmacy: " + err.message, "warn");
      }
    };

    // ---------- Approve EMR (doctor) ----------
    btnApproveEmr.onclick = async () => {
      if (currentRole !== "doctor") {
        setStatus("Only doctors can perform this action.", "warn");
        return;
      }
      if (!patientVerified) {
        setStatus("EMR locked: verify patient face first.", "warn");
        return;
      }
      const pid = getPatientId();
      if (!pid) {
        setStatus("Enter a Patient ID first.", "warn");
        return;
      }
      if (!currentState) {
        setStatus("Run a workflow first before approving.", "warn");
        return;
      }

      const noteSummary =
        (currentState.note_summary || currentState.raw_transcript || "").trim();
      const symptoms = Array.isArray(currentState.symptoms) ? currentState.symptoms : [];
      const testsLines = (testsEditBox.value || "")
        .split("\\n")
        .map(t => t.trim())
        .filter(t => t.length > 0);
      const rxText = (rxEditBox.value || "").trim();
      if (!rxText) {
        setStatus("Prescription text is empty. Please review/edit before approving.", "warn");
        return;
      }

      setStatus("Saving approved consultation to EMR...", "info");
      try {
        const res = await fetch("/approve-emr", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            patient_id: pid,
            note_summary: noteSummary,
            symptoms: symptoms,
            suggested_tests: testsLines,
            draft_prescription: rxText
          })
        });
        if (!res.ok) {
          const t = await res.text();
          throw new Error("Backend error " + res.status + ": " + t);
        }
        const json = await res.json();
        const emrId = json.emr_record_id || "";
        lastApprovedEmrId = emrId;
        setStatus("Consultation approved and saved to EMR as " + emrId, "ok");
        if (emrId) {
          emrIdBox.innerHTML =
            "<span style='font-size:0.75rem;color:#4ade80;'>Approved EMR stored as <code>" +
            emrId +
            "</code></span>";
        }
      } catch (err) {
        console.error(err);
        setStatus("Error saving EMR: " + err.message, "warn");
      }
    };
    btnLogin.onclick = () => {
      const selectedRole = roleSelect.value;
      const uname = (usernameInput.value || "").trim();
      const pwd = (passwordInput.value || "").trim();
      const loginPid = (loginPatientIdInput.value || "").trim();

      const user = USERS.find(
        u => u.username === uname && u.password === pwd && u.role === selectedRole
      );

      if (!user) {
        setStatus("Invalid credentials for selected role.", "warn");
        return;
      }

      if (selectedRole === "doctor" || selectedRole === "pharmacy") {
        if (!loginPid) {
          setStatus("Please enter a Patient ID for doctor / pharmacy login.", "warn");
          return;
        }
      }

      currentUser = user;
      currentRole = user.role;

      // Bind patient ID based on role
      if (currentRole === "patient" && user.patient_id) {
        // Patient: locked to their own ID
        patientIdInput.value = user.patient_id;
        patientIdInput.disabled = true;
      } else if (currentRole === "doctor" || currentRole === "pharmacy") {
        // Doctor / pharmacy: use patient ID from login
        if (loginPid) {
          patientIdInput.value = loginPid;
        }
        patientIdInput.disabled = false;
      }

      // Clear login form (optional)
      passwordInput.value = "";
      loginPatientIdInput.value = "";

      btnLogout.style.display = "inline-flex";

      updateRoleUI();

      // Auto-load relevant data
      if (currentRole === "patient" && btnPatientLoadEmr) {
        btnPatientLoadEmr.click();
      } else if (currentRole === "pharmacy" && btnPharmacyRefresh) {
        btnPharmacyRefresh.click();
      }
    };

    btnLogout.onclick = () => {
      currentUser = null;
      currentRole = null;
      usernameInput.value = "";
      passwordInput.value = "";
      loginPatientIdInput.value = "";
      patientIdInput.disabled = false;

      btnLogout.style.display = "none";

      setStatus("Logged out. Please login.", "info");
      updateRoleUI();
    };

    
    // ---------- Init ----------
    updateStepUI();
    setStatus("Step 1: Verify patient presence using camera. Then move to Listening Agents.", "info");
    updateRoleUI();
  </script>

</body>
</html>
    """

def check_doctor_allowed(db, patient_db_id: int, doctor_username: str) -> bool:
    """
    Returns True if this doctor has been granted access
    to this patient's EHR by the patient portal.
    """
    record = (
        db.query(PatientDoctorAccess)
        .filter(
            PatientDoctorAccess.patient_id == patient_db_id,
            PatientDoctorAccess.doctor_username == doctor_username,
        )
        .first()
    )
    return bool(record and record.is_allowed)

@app.post("/stt-only")
async def stt_only(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "")[1] or ".wav"
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file_bytes = await audio.read()
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        transcript = tool_transcribe_voice(tmp_path)
        return {"transcript": transcript}
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

@app.post("/approve-emr")
def approve_emr(req: ApproveEMRRequest):
    """
    Human-in-the-loop approval endpoint.

    Now:
    - Enforces biometric gate (face verification)
    - Persists a full Encounter into the EHR DB (encounters table)
    - Still returns an emr_record_id for UI
    """
    # 1) Biometric gate
    if not is_patient_authorized(req.patient_id):
        raise HTTPException(
            status_code=403,
            detail="Patient face not verified. EMR is locked for this patient."
        )

    # 2) Write into SQLite EHR DB
    db = SessionLocal()
    try:
        patient = get_or_create_patient(db, req.patient_id)

        encounter_id = f"ENC-{int(datetime.utcnow().timestamp())}"

        encounter = Encounter(
            encounter_id=encounter_id,
            patient_id=patient.id,
            created_at=datetime.utcnow(),
            doctor_username="doc1",  # TODO: map from login later

            # Clinical data
            note_summary=req.note_summary,
            symptoms=req.symptoms,
            suggested_tests=req.suggested_tests,

            # For now we don't collect these in UI, but you can wire them later:
            problems=[],
            medications=[],
            vitals={},  # you can extend ApproveEMRRequest with vitals field
            past_medical_history=[],

            prescription=req.draft_prescription,
            approved_by_doctor=True,
        )

        db.add(encounter)
        db.commit()
        db.refresh(encounter)

        emr_record_id = encounter.encounter_id

    finally:
        db.close()

    # 3) (Optional) also store to old JSON EMR for backwards-compat UI
    payload = {
        "record_type": "approved_consultation",
        "patient_id": req.patient_id,
        "note_summary": req.note_summary,
        "symptoms": req.symptoms,
        "suggested_tests": req.suggested_tests,
        "draft_prescription": req.draft_prescription,
        "approved_by_doctor": True,
    }
    tool_update_emr(payload)  # your existing mock EMR tool

    return {"status": "ok", "emr_record_id": emr_record_id}

@app.get("/patient/access-list")
def get_access_list(patient_id: str):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            raise HTTPException(404, "No such patient")

        records = (
            db.query(PatientDoctorAccess)
            .filter(PatientDoctorAccess.patient_id == patient.id)
            .all()
        )

        out = []
        for r in records:
            out.append({
                "doctor_username": r.doctor_username,
                "is_allowed": r.is_allowed
            })

        return {"status": "ok", "access_list": out}
    finally:
        db.close()

@app.post("/doctor/request-access")
def doctor_request_access(patient_id: str, doctor_username: str):
    # (Later we store request logs — for now we simulate)
    return {
        "status": "ok",
        "message": f"Access request sent to patient {patient_id}"
    }

@app.get("/ehr/{patient_id}")
def get_full_ehr(
    patient_id: str,
    role: str = Query("doctor"),      # "doctor" | "patient" | "pharmacy"
    username: str | None = Query(None),
):
    db = SessionLocal()
    try:
        patient = db.query(Patient).filter(Patient.patient_id == patient_id).first()
        if not patient:
            return {
                "patient_id": patient_id,
                "exists": False,
                "message": "No such patient in EHR.",
            }
        if not username:
            raise HTTPException(
                status_code=400,
                detail="username query param required (for demo access control).",
            )

        # 2) Patient portal: can only see their own EHR
        if role == "patient":
            pass
        elif role == "doctor":
            # (A) Face verification check (your existing gate)
            if not is_patient_authorized(patient_id):
                raise HTTPException(
                    status_code=403,
                    detail="Patient face not verified. EHR locked.",
                )
            if not check_doctor_allowed(db, patient.id, username):
                raise HTTPException(
                    status_code=403,
                    detail="Patient has not granted you access to this EHR.",
                )

        # 4) Pharmacy: we could restrict to pharmacy_orders only (later).
        elif role == "pharmacy":
            # For now we let it pass; in future, you can trim the payload.
            pass
        demo = {
            "patient_id": patient.patient_id,
            "full_name": patient.full_name,
            "date_of_birth": patient.date_of_birth.isoformat() if patient.date_of_birth else None,
            "gender": patient.gender,
            "phone": patient.phone,
            "email": patient.email,
            "address": patient.address,
            "emergency_contact_name": patient.emergency_contact_name,
            "emergency_contact_phone": patient.emergency_contact_phone,
            "blood_group": patient.blood_group,
            "allergies": patient.allergies or [],
            "chronic_conditions": patient.chronic_conditions or [],
        }

        # Insurance
        ins = None
        if patient.insurance_profile:
            ins = {
                "provider_name": patient.insurance_profile.provider_name,
                "policy_number": patient.insurance_profile.policy_number,
                "coverage_details": patient.insurance_profile.coverage_details,
                "billing_notes": patient.insurance_profile.billing_notes,
            }

        # Encounters
        encounters_out = []
        for enc in sorted(patient.encounters, key=lambda e: e.created_at or datetime.min, reverse=True):
            encounters_out.append({
                "encounter_id": enc.encounter_id,
                "created_at": enc.created_at.isoformat() if enc.created_at else None,
                "doctor_username": enc.doctor_username,
                "chief_complaint": enc.chief_complaint,
                "visit_type": enc.visit_type,
                "note_summary": enc.note_summary,
                "symptoms": enc.symptoms,
                "suggested_tests": enc.suggested_tests,
                "vitals": enc.vitals,
                "problems": enc.problems,
                "medications": enc.medications,
                "past_medical_history": enc.past_medical_history,
                "prescription": enc.prescription,
                "approved_by_doctor": enc.approved_by_doctor,
            })

        # Lab results
        labs_out = []
        for lab in patient.lab_results:
            labs_out.append({
                "id": lab.id,
                "test_name": lab.test_name,
                "result_value": lab.result_value,
                "unit": lab.unit,
                "reference_range": lab.reference_range,
                "status": lab.status,
                "report_text": lab.report_text,
                "encounter_id": lab.encounter.encounter_id if lab.encounter else None,
            })

        # Radiology reports
        rads_out = []
        for r in patient.radiology_reports:
            rads_out.append({
                "id": r.id,
                "modality": r.modality,
                "body_part": r.body_part,
                "impression": r.impression,
                "report_text": r.report_text,
                "encounter_id": r.encounter.encounter_id if r.encounter else None,
            })

        # Pharmacy orders
        orders_out = []
        for o in patient.pharmacy_orders:
            orders_out.append({
                "order_id": o.order_id,
                "created_at": o.created_at.isoformat() if o.created_at else None,
                "prescription": o.prescription,
                "status": o.status,
                "encounter_id": o.encounter_id,
            })

        return {
            "patient_id": patient.patient_id,
            "exists": True,
            "demographics": demo,
            "insurance": ins,
            "encounters": encounters_out,
            "lab_results": labs_out,
            "radiology_reports": rads_out,
            "pharmacy_orders": orders_out,
        }
    finally:
        db.close()

@app.post("/send-to-pharmacy")
def send_to_pharmacy(req: PharmacySendRequest):
    if not is_patient_authorized(req.patient_id):
        raise HTTPException(
            status_code=403,
            detail="Patient face not verified. Pharmacy actions are locked for this patient."
        )
    result = tool_send_to_pharmacy(
        {
            "patient_id": req.patient_id,
            "prescription": req.prescription,
            "emr_record_id": req.emr_record_id,
            "suggested_tests": req.suggested_tests,
            "symptoms": req.symptoms,
        }
    )
    order_id = (
        result.get("order_id")
        or result.get("pharmacy_order_id")
        or result.get("pharmacy_order_id".upper())
    )
    ts = result.get("timestamp_utc") or result.get("timestamp") or datetime.now(timezone.utc).isoformat()

    return {
        "status": result.get("status", "ok"),
        "order_id": order_id,
        "timestamp_utc": ts,
    }

@app.post("/enroll-patient-face")
async def enroll_patient_face(patient_id: str, image: UploadFile = File(...)):
    """
    Enrollment endpoint:
    - Called once per patient to register their reference face template.
    - Uses OpenCV + Haar cascade + simple template matching.
    """
    data = await image.read()
    try:
        info = enroll_from_image_bytes(patient_id, data)
        return {
            "status": "ok",
            "patient_id": patient_id,
            "template_path": info["template_path"],
            "shape": info["shape"],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/verify-patient-face")
async def verify_patient_face(patient_id: str, image: UploadFile = File(...)):
    """
    Verification endpoint:
    - Takes a live frame from the doctor's browser
    - Compares it to stored face template for this patient
    - On success: calls authorize_patient(patient_id)
    """
    data = await image.read()

    result = verify_from_image_bytes(patient_id, data)

    status = result.get("status")

    if status == "no_enrollment":
        raise HTTPException(
            status_code=404,
            detail="No enrolled face found for this patient. Please enroll first."
        )
    if status in ("decode_error", "no_face"):
        return {
            "authorized": False,
            "reason": f"Verification failed: {status}",
            "distance": result.get("distance"),
            "threshold": result.get("threshold"),
        }

    if not result["match"]:
        dist = result.get("distance")
        return {
            "authorized": False,
            "reason": f"Face mismatch (distance={dist:.3f}, threshold={result['threshold']:.3f})",
            "distance": dist,
            "threshold": result["threshold"],
        }

    # ✅ Face matched – unlock EMR & workflow for this patient
    authorize_patient(patient_id)

    dist = result.get("distance")
    return {
        "authorized": True,
        "reason": f"Face matched (distance={dist:.3f}, threshold={result['threshold']:.3f})",
        "distance": dist,
        "threshold": result["threshold"],
    }


@app.get("/qdrant-debug", response_class=HTMLResponse)
def qdrant_debug_page():
    """
    Minimal web UI to inspect Qdrant collections & points.
    """
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Qdrant Debug Viewer</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #020617;
      color: #e5e7eb;
      margin: 0;
      padding: 16px;
    }
    h1 {
      font-size: 1.4rem;
      color: #38bdf8;
      margin-bottom: 8px;
    }
    .layout {
      display: grid;
      grid-template-columns: 260px 1fr;
      gap: 16px;
    }
    .card {
      background: #020617;
      border: 1px solid #1f2937;
      border-radius: 12px;
      padding: 10px 12px;
      box-shadow: 0 14px 30px rgba(0,0,0,0.6);
    }
    ul {
      list-style: none;
      margin: 0;
      padding: 0;
      max-height: 320px;
      overflow-y: auto;
      font-size: 0.85rem;
    }
    li {
      padding: 6px 8px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    li:hover {
      background: #0f172a;
    }
    li.active {
      background: #0f172a;
      border: 1px solid #38bdf8;
    }
    .pill {
      font-size: 0.75rem;
      color: #9ca3af;
    }
    pre {
      background: #020617;
      border-radius: 8px;
      border: 1px solid #1f2937;
      padding: 8px;
      font-size: 0.75rem;
      max-height: 360px;
      overflow: auto;
      white-space: pre-wrap;
    }
    .point {
      border-bottom: 1px solid #1f2937;
      padding-bottom: 6px;
      margin-bottom: 6px;
    }
    .point:last-child {
      border-bottom: none;
    }
    code {
      background: #020617;
      padding: 2px 4px;
      border-radius: 4px;
      font-size: 0.75rem;
      color: #93c5fd;
    }
    select, input {
      background: #020617;
      color: #e5e7eb;
      border-radius: 999px;
      border: 1px solid #1f2937;
      padding: 4px 8px;
      font-size: 0.75rem;
      outline: none;
    }
    button {
      border-radius: 999px;
      border: none;
      padding: 4px 10px;
      font-size: 0.75rem;
      cursor: pointer;
      background: #06b6d4;
      color: #020617;
      font-weight: 600;
      box-shadow: 0 8px 18px rgba(8,145,178,0.7);
      margin-left: 6px;
    }
  </style>
</head>
<body>
  <h1>Qdrant Debug Viewer</h1>
  <p style="font-size:0.8rem;color:#9ca3af;">
    Inspect your local Qdrant collections and see stored guideline chunks / vectors.
  </p>
  <div class="layout">
    <div class="card">
      <h2 style="font-size:1rem;margin:0 0 6px;">Collections</h2>
      <button id="btnReloadCols">Reload</button>
      <ul id="collectionList"></ul>
    </div>
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h2 id="colTitle" style="font-size:1rem;margin:0;">No collection selected</h2>
        <div>
          <span class="pill">Limit</span>
          <input id="limitInput" type="number" value="20" min="1" max="200" style="width:60px;">
          <button id="btnReloadPoints">Go</button>
        </div>
      </div>
      <div id="pointsBox" style="margin-top:8px;font-size:0.8rem;color:#e5e7eb;">
        <p style="color:#9ca3af;font-size:0.8rem;">Select a collection to inspect its points.</p>
      </div>
    </div>
  </div>

  <script>
    const collectionList = document.getElementById("collectionList");
    const btnReloadCols = document.getElementById("btnReloadCols");
    const colTitle = document.getElementById("colTitle");
    const pointsBox = document.getElementById("pointsBox");
    const limitInput = document.getElementById("limitInput");
    const btnReloadPoints = document.getElementById("btnReloadPoints");

    let currentCollection = null;

    async function loadCollections() {
      collectionList.innerHTML = "<li><span class='pill'>Loading collections...</span></li>";
      try {
        const res = await fetch("/qdrant/collections");
        if (!res.ok) {
          const t = await res.text();
          throw new Error(t);
        }
        const data = await res.json();
        const cols = data.collections || [];
        collectionList.innerHTML = "";
        if (cols.length === 0) {
          collectionList.innerHTML = "<li><span class='pill'>No collections found.</span></li>";
          return;
        }
        cols.forEach(c => {
          const li = document.createElement("li");
          li.dataset.name = c.name;
          li.innerHTML =
            "<span>" + c.name + "</span>" +
            "<span class='pill'>" + (c.vectors_count ?? "?") + " pts</span>";
          li.onclick = () => {
            currentCollection = c.name;
            document.querySelectorAll("#collectionList li").forEach(el => el.classList.remove("active"));
            li.classList.add("active");
            loadPoints();
          };
          collectionList.appendChild(li);
        });
      } catch (err) {
        collectionList.innerHTML = "<li><span class='pill'>Error: " + err.message + "</span></li>";
      }
    }

    async function loadPoints() {
      if (!currentCollection) {
        pointsBox.innerHTML = "<p style='color:#9ca3af;'>No collection selected.</p>";
        return;
      }
      colTitle.textContent = "Collection: " + currentCollection;
      const limit = parseInt(limitInput.value || "20", 10) || 20;
      pointsBox.innerHTML = "<p style='color:#9ca3af;'>Loading points...</p>";
      try {
        const res = await fetch("/qdrant/collection/" + encodeURIComponent(currentCollection) + "?limit=" + limit);
        if (!res.ok) {
          const t = await res.text();
          throw new Error(t);
        }
        const data = await res.json();
        const pts = data.points || [];
        if (pts.length === 0) {
          pointsBox.innerHTML = "<p style='color:#9ca3af;'>No points in this collection.</p>";
          return;
        }
        pointsBox.innerHTML = "";
        pts.forEach(p => {
          const div = document.createElement("div");
          div.className = "point";
          const keys = (p.payload_keys || []).join(", ");
          const preview = p.preview || "";
          div.innerHTML =
            "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'>" +
            "<span style='font-family:monospace;color:#38bdf8;font-size:0.8rem;'>ID: " + p.id + "</span>" +
            "<span class='pill'>Payload: " + keys + "</span>" +
            "</div>" +
            (preview
              ? "<pre>" + preview + "</pre>"
              : "<p style='color:#9ca3af;font-size:0.75rem;'>No text preview.</p>") +
            "<details style='margin-top:4px;'>" +
            "<summary style='font-size:0.75rem;color:#38bdf8;cursor:pointer;'>Full payload</summary>" +
            "<pre>" + JSON.stringify(p.payload, null, 2) + "</pre>" +
            "</details>";
          pointsBox.appendChild(div);
        });
      } catch (err) {
        pointsBox.innerHTML = "<p style='color:#f97373;'>Error loading points: " + err.message + "</p>";
      }
    }

    btnReloadCols.onclick = loadCollections;
    btnReloadPoints.onclick = loadPoints;

    // initial load
    loadCollections();
  </script>
</body>
</html>
    """
