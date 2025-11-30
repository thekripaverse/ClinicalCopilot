# backend/nodes/planner_node.py

from typing import List, Dict, Any

from ..state import AgentState
from ..tools import rag_query_tool

"""
Planner Node

Responsibilities:
- Look at symptoms + note_summary text
- Use STANDARD_TESTS (rule-based) to propose investigations
- Optionally enrich using RAG (Qdrant) guideline snippets
- Fill:
    - state.suggested_tests
    - state.guideline_hits
    - audit_log
"""

# -------------------------------
# Standard Professional Test Map
# -------------------------------

STANDARD_TESTS: Dict[str, List[str]] = {
    "chest pain": [
        "ECG 12-lead",
        "Cardiac Troponin I",
        "CK-MB",
        "Lipid Profile",
        "Chest X-Ray (PA view)",
        "Serum Electrolytes",
        "Echocardiogram",
    ],
    "shortness of breath": [
        "Chest X-Ray (PA view)",
        "Arterial Blood Gas (ABG)",
        "D-Dimer",
        "CBC with Differential",
        "HRCT Thorax",
        "NT-proBNP",
    ],
    "fever": [
        "CBC with Differential",
        "C-Reactive Protein (CRP)",
        "ESR",
        "Dengue NS1 / IgM",
        "Malaria Smear / Rapid Test",
        "Blood Culture",
        "Urine Routine and Microscopy",
        "Procalcitonin",
    ],
    "cough": [
        "Chest X-Ray (PA view)",
        "CBC with Differential",
        "CRP",
        "Sputum Culture and Sensitivity",
        "COVID-19 RT-PCR",
        "Sputum AFB (for Tuberculosis)",
    ],
    "headache": [
        "MRI Brain (as indicated)",
        "CT Brain (Non-contrast, if emergency)",
        "CBC",
        "Serum Electrolytes",
        "ESR",
    ],
    "vomiting": [
        "CBC",
        "Serum Electrolytes",
        "Random Blood Sugar",
        "Serum Amylase",
        "Serum Lipase",
        "Liver Function Test (LFT)",
        "Ultrasound Abdomen",
    ],
    "abdominal pain": [
        "CBC",
        "Liver Function Test (LFT)",
        "Serum Amylase",
        "Serum Lipase",
        "Ultrasound Abdomen",
        "Stool Routine and Occult Blood",
    ],
    "diabetes": [
        "HbA1c",
        "Fasting Plasma Glucose",
        "Postprandial Blood Sugar",
        "Lipid Profile",
        "Urine Microalbumin",
        "Renal Function Test (RFT)",
    ],
    "hypertension": [
        "Serum Electrolytes",
        "ECG 12-lead",
        "Echocardiogram",
        "Renal Doppler Ultrasound",
        "Urine Routine and Microscopy",
    ],
}


def _tests_from_symptoms(symptoms: List[str]) -> List[str]:
    """
    Expand structured symptoms into a consolidated list of
    recommended tests using STANDARD_TESTS.
    """
    tests: List[str] = []
    for symptom in symptoms:
        key = symptom.lower()
        if key in STANDARD_TESTS:
            tests.extend(STANDARD_TESTS[key])
    return list(dict.fromkeys(tests))  # unique, ordered


def _tests_from_text(note_text: str) -> List[str]:
    """
    Fallback: directly scan note text for symptom phrases and
    suggest tests even if symptom_node missed them.
    """
    text = (note_text or "").lower()
    detected_symptoms: List[str] = []

    # Very similar mapping to symptom_node, but inline
    phrase_to_symptom = {
        # chest pain
        "chest pain": "chest pain",
        "pain in chest": "chest pain",
        "heaviness in chest": "chest pain",
        "tightness in chest": "chest pain",
        "pressure in chest": "chest pain",
        # shortness of breath
        "shortness of breath": "shortness of breath",
        "breathlessness": "shortness of breath",
        "breathless": "shortness of breath",
        "difficulty breathing": "shortness of breath",
        # fever
        "fever": "fever",
        "high temperature": "fever",
        # cough
        "cough": "cough",
        "coughing": "cough",
        # headache
        "headache": "headache",
        "pain in head": "headache",
        "migraine": "headache",
        # vomiting
        "vomiting": "vomiting",
        "vomit": "vomiting",
        "threw up": "vomiting",
        "nausea": "vomiting",
        # abdominal pain
        "abdominal pain": "abdominal pain",
        "stomach pain": "abdominal pain",
        "tummy pain": "abdominal pain",
        "gastric pain": "abdominal pain",
        # diabetes
        "diabetes": "diabetes",
        "type 2 diabetes": "diabetes",
        "type ii diabetes": "diabetes",
        "high blood sugar": "diabetes",
        # hypertension
        "hypertension": "hypertension",
        "high blood pressure": "hypertension",
        "bp is high": "hypertension",
    }

    for phrase, canonical in phrase_to_symptom.items():
        if phrase in text:
            detected_symptoms.append(canonical)

    detected_symptoms = list(dict.fromkeys(detected_symptoms))

    # Now just reuse STANDARD_TESTS for these
    return _tests_from_symptoms(detected_symptoms)


def _tests_from_rag(hits: List[Dict[str, Any]]) -> List[str]:
    """
    Extract possible tests from guideline texts returned by RAG.
    Very simple keyword-based extractor for demo purposes.
    """
    candidate_keywords = [
        "cbc",
        "ecg",
        "x-ray",
        "xray",
        "chest x-ray",
        "abg",
        "d-dimer",
        "lipid",
        "urine",
        "hba1c",
        "glucose",
        "ct",
        "mri",
        "ultrasound",
        "spiro",
        "spirometry",
        "rft",
        "lft",
        "esr",
        "crp",
        "procalcitonin",
        "troponin",
    ]

    extracted: List[str] = []

    for h in hits:
        text = (h.get("text") or "").lower()
        for kw in candidate_keywords:
            if kw in text:
                # Normalize to nice display names
                if kw == "cbc":
                    name = "CBC with Differential"
                elif kw == "ecg":
                    name = "ECG 12-lead"
                elif kw in ["x-ray", "xray", "chest x-ray"]:
                    name = "Chest X-Ray (PA view)"
                elif kw == "abg":
                    name = "Arterial Blood Gas (ABG)"
                elif kw == "lipid":
                    name = "Lipid Profile"
                elif kw == "urine":
                    name = "Urine Routine and Microscopy"
                elif kw == "hba1c":
                    name = "HbA1c"
                elif kw == "glucose":
                    name = "Fasting / Random Blood Glucose"
                elif kw == "ct":
                    name = "CT Scan (site as clinically indicated)"
                elif kw == "mri":
                    name = "MRI (site as clinically indicated)"
                elif kw == "ultrasound":
                    name = "Ultrasound (region as clinically indicated)"
                elif kw in ["spiro", "spirometry"]:
                    name = "Spirometry (Pulmonary Function Test)"
                elif kw == "rft":
                    name = "Renal Function Test (RFT)"
                elif kw == "lft":
                    name = "Liver Function Test (LFT)"
                elif kw == "esr":
                    name = "ESR"
                elif kw == "crp":
                    name = "C-Reactive Protein (CRP)"
                elif kw == "procalcitonin":
                    name = "Procalcitonin"
                elif kw == "troponin":
                    name = "Cardiac Troponin I"
                else:
                    name = kw.upper()

                if name not in extracted:
                    extracted.append(name)

    return extracted


def planner_node(state: AgentState) -> AgentState:
    """
    Main planner:

    1. Use structured symptoms (from symptom_node) to get a
       base set of standard tests.
    2. Fallback: scan note text directly and infer tests.
    3. Call RAG (Qdrant) to fetch guideline snippets and extract more tests.
    4. Merge & dedupe into state.suggested_tests.
    """
    note = state.note_summary or state.raw_transcript or ""
    symptoms_text = ", ".join(state.symptoms) if state.symptoms else "unspecified symptoms"

    # 1) Rule-based tests from structured symptoms
    rule_tests_from_symptoms = _tests_from_symptoms([s.lower() for s in state.symptoms])

    # 2) Fallback: tests inferred directly from note text
    rule_tests_from_text = _tests_from_text(note)

    # 3) RAG call (safe: rag_query_tool returns [] on failure)
    query = f"Suggest initial investigations for a patient with: {symptoms_text}. Note: {note}"
    hits = rag_query_tool(query, top_k=3)
    rag_tests = _tests_from_rag(hits)

    # 4) Merge all
    combined: List[str] = []
    for t in rule_tests_from_symptoms + rule_tests_from_text + rag_tests:
        if t not in combined:
            combined.append(t)

    state.guideline_hits = hits
    state.suggested_tests = combined

    state.audit_log.append(
        "Planner node: "
        f"symptoms={state.symptoms}, "
        f"rule_from_symptoms={rule_tests_from_symptoms}, "
        f"rule_from_text={rule_tests_from_text}, "
        f"rag_tests={rag_tests}, "
        f"final_suggested_tests={combined}."
    )

    return state
