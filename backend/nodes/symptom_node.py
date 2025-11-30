# backend/nodes/symptom_node.py

from typing import List
from ..state import AgentState

"""
Symptom Extraction Agent

Reads note_summary/raw_transcript and fills:
    state.symptoms = [canonical_symptom_1, ...]
"""

PHRASE_TO_SYMPTOM = {
    # Chest pain
    "chest pain": "chest pain",
    "pain in chest": "chest pain",
    "heaviness in chest": "chest pain",
    "tightness in chest": "chest pain",
    "pressure in chest": "chest pain",
    "chest discomfort": "chest pain",

    # Shortness of breath
    "shortness of breath": "shortness of breath",
    "sob": "shortness of breath",
    "breathlessness": "shortness of breath",
    "breathless": "shortness of breath",
    "difficulty breathing": "shortness of breath",
    "cannot breathe": "shortness of breath",
    "wheezing": "shortness of breath",

    # Fever
    "fever": "fever",
    "high temperature": "fever",
    "running temperature": "fever",
    "felt hot": "fever",

    # Cough
    "cough": "cough",
    "coughing": "cough",
    "dry cough": "cough",
    "productive cough": "cough",

    # Headache
    "headache": "headache",
    "pain in head": "headache",
    "migraine": "headache",
    "throbbing headache": "headache",

    # Vomiting / nausea
    "vomiting": "vomiting",
    "vomit": "vomiting",
    "threw up": "vomiting",
    "throwing up": "vomiting",
    "nausea": "vomiting",
    "feel like vomiting": "vomiting",

    # Abdominal pain
    "abdominal pain": "abdominal pain",
    "stomach pain": "abdominal pain",
    "tummy pain": "abdominal pain",
    "gastric pain": "abdominal pain",
    "pain in abdomen": "abdominal pain",

    # Diarrhoea
    "loose stools": "diarrhoea",
    "loose motions": "diarrhoea",
    "diarrhea": "diarrhoea",
    "diarrhoea": "diarrhoea",

    # Constipation
    "constipation": "constipation",
    "hard stools": "constipation",
    "difficulty passing motion": "constipation",

    # UTI / urinary symptoms
    "burning urination": "uti",
    "burning while passing urine": "uti",
    "painful urination": "uti",
    "frequent urination": "uti",
    "urine frequency": "uti",
    "blood in urine": "uti",

    # Dizziness / giddiness
    "dizziness": "dizziness",
    "giddiness": "dizziness",
    "feel like spinning": "dizziness",
    "vertigo": "dizziness",
    "light headed": "dizziness",

    # Syncope / fainting
    "fainted": "syncope",
    "lost consciousness": "syncope",
    "passed out": "syncope",
    "blackout": "syncope",

    # Palpitations
    "palpitations": "palpitations",
    "heart racing": "palpitations",
    "heart beating fast": "palpitations",
    "pounding heart": "palpitations",

    # Leg swelling / edema
    "leg swelling": "leg edema",
    "swollen legs": "leg edema",
    "pedal edema": "leg edema",
    "ankle swelling": "leg edema",

    # Neurological red flags
    "weakness of one side": "focal weakness",
    "weakness on one side": "focal weakness",
    "slurred speech": "slurred speech",
    "difficulty speaking": "slurred speech",
    "sudden vision loss": "visual loss",

    # Diabetes / high sugar
    "diabetes": "diabetes",
    "type 2 diabetes": "diabetes",
    "type ii diabetes": "diabetes",
    "high blood sugar": "diabetes",
    "sugar is high": "diabetes",

    # Hypertension / high bp
    "hypertension": "hypertension",
    "high blood pressure": "hypertension",
    "bp is high": "hypertension",
    "pressure is high": "hypertension",

    # Anemia
    "anemia": "anemia",
    "anaemia": "anemia",
    "low hemoglobin": "anemia",
    "hb is low": "anemia",

    # Pregnancy
    "pregnant": "pregnancy",
    "missed period": "pregnancy",
    "positive pregnancy test": "pregnancy",
}


def extract_symptoms_from_text(text: str) -> List[str]:
    text = (text or "").lower()
    detected = []
    for phrase, canonical in PHRASE_TO_SYMPTOM.items():
        if phrase in text:
            detected.append(canonical)
    # make unique, preserve order
    return list(dict.fromkeys(detected))


def symptom_node(state: AgentState) -> AgentState:
    note = state.note_summary or state.raw_transcript or ""
    symptoms = extract_symptoms_from_text(note)
    state.symptoms = symptoms
    state.audit_log.append(f"Symptom node: extracted symptoms {symptoms}.")
    return state
