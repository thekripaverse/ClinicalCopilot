# backend/auth.py

AUTHORIZED_PATIENTS = set()

def authorize_patient(patient_id: str):
    AUTHORIZED_PATIENTS.add(patient_id)

def is_patient_authorized(patient_id: str) -> bool:
    return patient_id in AUTHORIZED_PATIENTS

def revoke_patient(patient_id: str):
    AUTHORIZED_PATIENTS.discard(patient_id)

def clear_all():
    AUTHORIZED_PATIENTS.clear()
