# ClinicalCopilot


This project is an **agentic healthcare automation system** that orchestrates end-to-end clinical workflows using **multimodal AI**, **vector-based medical retrieval**, and **secure EMR actions**.
It reduces physician administrative load by automating listening, summarization, symptom extraction, investigation planning, prescription drafting, and pharmacy order dispatch.

The system integrates:

###  **1. Biometric Patient Verification**

* Live camera feed + facial recognition checkpoint
* Locks all clinical actions until patient identity is confirmed
* Prevents unauthorized EMR or prescription access
* Acts as a “Presence Agent” in the workflow

---

###  **2. Multimodal Listening Agent**

* Supports **live microphone streaming** + **WAV audio upload**
* Uses robust **system-level speech-to-text** (non-cloud)
* Converts consultation speech → structured transcript
* No dependence on paid APIs

---

###  **3. Medical Reasoning Agents**

A chain of domain-specific agents built as modular nodes:

* **Scribe Agent** → cleans transcript, produces clinical note
* **Symptom Extraction Agent** → maps lay language → canonical symptoms
* **Planner Agent** → recommends investigations via

  * rule-based mapping
  * **local Qdrant vector DB** + guideline retrieval
* **Prescription Agent** → drafts structured provisional medication plan
* **Safety Agent** → identifies emergency red flags

These nodes form an **Agentic Workflow Graph** orchestrated by the backend.

---

###  **4. RAG-Driven Clinical Insights (Local Qdrant)**

* Local **Qdrant Vector DB** stores medical guideline chunks
* Planner agent performs guideline-aware test selection
* No external LLMS used
* Fully offline & reproducible

---

###  **5. EMR Integration Layer**

* After doctor review, approved consultations are persisted into:
  **`emr_store.json` → mock EMR system**
* All EMR actions enforce biometric gate
* Every state transition is logged in the audit trail

---

###  **6. Pharmacy Action Agent**

* Converts approved prescriptions → pharmacy orders
* Writes orders to **`pharmacy_orders.json`**
* Includes EMR record linkage, timestamping, and action metadata
* Demonstrates real-world “agent → external tool” interoperability

---

###  **7. Unified Clinical Dashboard (Local UI)**

The backend serves an interactive, agent-aware dashboard with:

* Face verification panel
* Audio/Livemic clinical capture
* Editable transcript
* Auto-generated symptoms, tests, prescription
* EMR records viewer
* Pharmacy orders viewer
* Audit timeline for explainability

Everything is rendered using **pure HTML/CSS/JS** inside FastAPI—no external frontend frameworks needed.

---

# Core Capabilities

* **Agentic workflow automation** for clinical operations
* **Secure patient-gated EMR write access**
* **Healthcare RAG engine using local vector database**
* **Multimodal STT ingestion pipeline**
* **Explainable AI with step-by-step audit logs**
* **End-to-end consultation → EMR → Pharmacy flow**
* **Completely offline, no paid cloud APIs**
* **Modular nodes enabling extensibility (labs, insurance, billing, alerts)**

