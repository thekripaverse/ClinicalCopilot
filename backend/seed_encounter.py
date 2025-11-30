from datetime import datetime, timedelta
from typing import List, Dict, Any

from db import (
    SessionLocal,
    Patient,
    Encounter,
    LabResult,
    RadiologyReport,
    PharmacyOrder,
)


def make_vitals(bp: str, pulse: int, temp_c: float, spo2: int) -> Dict[str, Any]:
    return {
        "blood_pressure": bp,
        "pulse": pulse,
        "temperature_c": temp_c,
        "spo2": spo2,
    }


def seed_for_patient(db, patient: Patient, base_idx: int = 1):
    """
    Create 1–2 realistic encounters + labs + radiology + pharmacy
    for a single patient row.
    """

    pid = patient.patient_id
    print(f"Seeding EHR data for patient {pid} ({patient.full_name or 'no-name'})")

    # You can fine-tune per patient ID; for demo we branch by suffix
    scenario_type = int(pid[-1])  # last digit 1..0

    encounters: List[Encounter] = []

    now = datetime.utcnow()

    # ---- Scenario definitions ----
    if scenario_type in (1, 2):  # Fever / URI scenario
        enc = Encounter(
            encounter_id=f"ENC-{pid}-001",
            patient_id=patient.id,
            created_at=now - timedelta(days=3),
            doctor_username="doc1",
            chief_complaint="Fever and cough for 3 days",
            visit_type="OPD",

            note_summary=(
                "Patient presents with low-grade fever, dry cough and sore throat "
                "for 3 days. No breathlessness at rest. No prior similar episodes."
            ),
            problems=["Acute upper respiratory infection"],
            medications=["Paracetamol 500 mg TID for 3 days", "Cough syrup PRN"],
            vitals=make_vitals(bp="118/76", pulse=92, temp_c=38.2, spo2=98),
            past_medical_history=["No known chronic illnesses"],

            symptoms=["fever", "cough", "sore throat"],
            suggested_tests=["CBC with Differential", "CRP", "COVID-19 RT-PCR"],

            prescription=(
                "Tab Paracetamol 500 mg PO TID x 3 days\n"
                "Cough syrup 10 ml PO TID PRN\n"
                "Steam inhalation BID\n"
            ),
            approved_by_doctor=True,
        )
        encounters.append(enc)

    elif scenario_type in (3, 4):  # Diabetes / hypertension follow-up
        enc = Encounter(
            encounter_id=f"ENC-{pid}-001",
            patient_id=patient.id,
            created_at=now - timedelta(days=10),
            doctor_username="doc1",
            chief_complaint="Diabetes and hypertension follow-up",
            visit_type="Follow-up",

            note_summary=(
                "Known case of type 2 diabetes mellitus and hypertension for 5 years. "
                "Presents for routine follow-up. Complains of occasional dizziness. "
                "Home BP readings mildly elevated."
            ),
            problems=["Type 2 diabetes mellitus", "Essential hypertension"],
            medications=[
                "Metformin 500 mg BID",
                "Amlodipine 5 mg OD",
            ],
            vitals=make_vitals(bp="142/90", pulse=84, temp_c=36.9, spo2=98),
            past_medical_history=["Type 2 diabetes mellitus", "Hypertension"],

            symptoms=["dizziness", "polyuria"],
            suggested_tests=[
                "Fasting Blood Sugar",
                "HbA1c",
                "Serum Creatinine",
                "Lipid Profile",
            ],

            prescription=(
                "Continue Metformin 500 mg BID\n"
                "Increase Amlodipine to 10 mg OD\n"
                "Lifestyle modification counselling given\n"
            ),
            approved_by_doctor=True,
        )
        encounters.append(enc)

    elif scenario_type in (5, 6):  # Chest pain evaluation
        enc = Encounter(
            encounter_id=f"ENC-{pid}-001",
            patient_id=patient.id,
            created_at=now - timedelta(days=1),
            doctor_username="doc1",
            chief_complaint="Intermittent chest pain for 1 day",
            visit_type="OPD",

            note_summary=(
                "Patient complains of intermittent central chest pain for 1 day, "
                "non-radiating, not clearly exertional. No associated sweating "
                "or shortness of breath. Risk factors: smoker, dyslipidemia."
            ),
            problems=["Chest pain - rule out ischemic heart disease"],
            medications=["Aspirin 75 mg OD", "Statin as per protocol"],
            vitals=make_vitals(bp="130/82", pulse=88, temp_c=36.8, spo2=99),
            past_medical_history=["Smoker", "Dyslipidemia"],

            symptoms=["chest pain", "anxiety"],
            suggested_tests=[
                "ECG 12 lead",
                "Cardiac Troponin I",
                "Chest X-Ray (PA view)",
                "Lipid Profile",
            ],

            prescription=(
                "Aspirin 75 mg OD\n"
                "High-intensity statin OD\n"
                "Advise ECG and cardiac enzymes urgently\n"
            ),
            approved_by_doctor=True,
        )
        encounters.append(enc)

    elif scenario_type in (7, 8):  # Asthma / SOB
        enc = Encounter(
            encounter_id=f"ENC-{pid}-001",
            patient_id=patient.id,
            created_at=now - timedelta(days=5),
            doctor_username="doc1",
            chief_complaint="Shortness of breath and wheeze",
            visit_type="OPD",

            note_summary=(
                "Known asthmatic with increasing wheeze over 5 days, using inhaler "
                "more frequently. No fever. Peak flow reduced from baseline."
            ),
            problems=["Bronchial asthma – acute exacerbation"],
            medications=[
                "Salbutamol inhaler PRN",
                "Inhaled corticosteroid BID",
            ],
            vitals=make_vitals(bp="124/80", pulse=104, temp_c=37.0, spo2=93),
            past_medical_history=["Bronchial asthma since childhood"],

            symptoms=["shortness of breath", "wheeze", "cough"],
            suggested_tests=[
                "Chest X-Ray (PA view)",
                "Spirometry",
                "CBC",
            ],

            prescription=(
                "Salbutamol inhaler 2 puffs q6h PRN\n"
                "ICS/LABA combination inhaler BID\n"
                "Review in 3 days or earlier if worsening\n"
            ),
            approved_by_doctor=True,
        )
        encounters.append(enc)

    else:  # scenario_type 9 or 0 – general checkup
        enc = Encounter(
            encounter_id=f"ENC-{pid}-001",
            patient_id=patient.id,
            created_at=now - timedelta(days=20),
            doctor_username="doc1",
            chief_complaint="General health check-up",
            visit_type="OPD",

            note_summary=(
                "Patient presents for routine annual health check. No major complaints. "
                "Non-smoker, moderate exercise, no chronic diseases known."
            ),
            problems=["Routine health check"],
            medications=[],
            vitals=make_vitals(bp="120/78", pulse=76, temp_c=36.7, spo2=99),
            past_medical_history=["No significant past medical history"],

            symptoms=["fatigue"],
            suggested_tests=[
                "CBC",
                "Lipid Profile",
                "Fasting Blood Sugar",
                "LFT",
            ],

            prescription=(
                "No regular medications started.\n"
                "Lifestyle counselling for diet and exercise.\n"
            ),
            approved_by_doctor=True,
        )
        encounters.append(enc)

    # Optionally add a second encounter with different time
    # You can expand this if you want multi-visit histories later.

    # ---- Persist encounters ----
    for enc in encounters:
        # Avoid duplicates if script run multiple times
        existing = (
            db.query(Encounter)
            .filter(Encounter.encounter_id == enc.encounter_id)
            .first()
        )
        if existing:
            print(f"  Encounter {enc.encounter_id} already exists, skipping.")
            continue

        db.add(enc)
        db.flush()  # we need enc.id for lab/radiology/pharmacy

        # Link Labo results based on scenario
        if "CBC" in (enc.suggested_tests or []):
            db.add(
                LabResult(
                    patient_id=patient.id,
                    encounter_id=enc.id,
                    test_name="CBC",
                    result_value="Mild leukocytosis",
                    unit="",
                    reference_range="WBC 4-11 x10^9/L",
                    status="completed",
                    report_text="Mildly elevated WBC suggesting infection.",
                )
            )

        if "HbA1c" in (enc.suggested_tests or []):
            db.add(
                LabResult(
                    patient_id=patient.id,
                    encounter_id=enc.id,
                    test_name="HbA1c",
                    result_value="7.8",
                    unit="%",
                    reference_range="< 7.0%",
                    status="completed",
                    report_text="Suboptimal glycemic control.",
                )
            )

        # Radiology
        if any("Chest X-Ray" in t for t in (enc.suggested_tests or [])):
            db.add(
                RadiologyReport(
                    patient_id=patient.id,
                    encounter_id=enc.id,
                    modality="X-ray",
                    body_part="Chest",
                    impression="No acute infiltrates.",
                    report_text="PA view chest X-ray shows clear lung fields, normal cardiac silhouette.",
                )
            )

        # Pharmacy order corresponding to prescription
        if enc.prescription:
            order_id = f"PHARM-{enc.encounter_id}"
            existing_order = (
                db.query(PharmacyOrder)
                .filter(PharmacyOrder.order_id == order_id)
                .first()
            )
            if not existing_order:
                db.add(
                    PharmacyOrder(
                        order_id=order_id,
                        patient_id=patient.id,
                        encounter_id=enc.id,
                        created_at=enc.created_at,
                        prescription=enc.prescription,
                        status="pending",
                    )
                )

    db.commit()
    print(f"  -> Seeded {len(encounters)} encounter(s) + labs/radiology/pharmacy\n")


def main():
    db = SessionLocal()
    try:
        # Fetch patients P001..P010
        patient_ids = [f"P{str(i).zfill(3)}" for i in range(1, 11)]
        patients = (
            db.query(Patient)
            .filter(Patient.patient_id.in_(patient_ids))
            .all()
        )
        found_ids = {p.patient_id for p in patients}
        missing = set(patient_ids) - found_ids
        if missing:
            print("WARNING: These patient_ids not found in DB:", sorted(missing))

        for p in patients:
            seed_for_patient(db, p)

    finally:
        db.close()


if __name__ == "__main__":
    main()
