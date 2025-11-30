from datetime import date, datetime
from db import (
    Base,
    engine,
    SessionLocal,
    Patient,
    InsuranceProfile,
    Encounter,
    LabResult,
    RadiologyReport,
    PharmacyOrder,
    PatientDoctorAccess,
)

# ---------------------------------------------------------------------
# Reset & create tables
# ---------------------------------------------------------------------
print("🔄 Resetting database...")
Base.metadata.drop_all(bind=engine)
Base.metadata.create_all(bind=engine)
print("✅ Tables created.")

db = SessionLocal()

# ---------------------------------------------------------------------
# 10 Demo Patients — Full EHR
# ---------------------------------------------------------------------

patients_data = [
    {
        "patient_id": "P001",
        "full_name": "kripasree",
        "dob": date(1988, 3, 12),
        "gender": "Female",
        "phone": "9876543210",
        "email": "arun@example.com",
        "address": "Coimbatore, Tamil Nadu",
        "emergency_name": "Priya Kumar",
        "emergency_phone": "9876509876",
        "blood_group": "O+",
        "allergies": ["Penicillin"],
        "chronic": ["Hypertension"],
    },
    {
        "patient_id": "P002",
        "full_name": "Madhu",
        "dob": date(1994, 7, 25),
        "gender": "Female",
        "phone": "9445566778",
        "email": "meera@example.com",
        "address": "Chennai, Tamil Nadu",
        "emergency_name": "Chandran",
        "emergency_phone": "9445532121",
        "blood_group": "A+",
        "allergies": [],
        "chronic": ["Asthma"],
    },
    {
        "patient_id": "P003",
        "full_name": "Nandita",
        "dob": date(2001, 11, 5),
        "gender": "Female",
        "phone": "9003001122",
        "email": "rohan@example.com",
        "address": "Bangalore, Karnataka",
        "emergency_name": "Sundar",
        "emergency_phone": "9003002211",
        "blood_group": "B+",
        "allergies": ["Seafood"],
        "chronic": [],
    },
    {
        "patient_id": "P004",
        "full_name": "preethi",
        "dob": date(1982, 1, 14),
        "gender": "Female",
        "phone": "9845012345",
        "email": "divya@example.com",
        "address": "Hyderabad, Telangana",
        "emergency_name": "Naveen",
        "emergency_phone": "9845098765",
        "blood_group": "AB+",
        "allergies": [],
        "chronic": ["Diabetes"],
    },
    {
        "patient_id": "P005",
        "full_name": "Jenilia",
        "dob": date(1990, 9, 9),
        "gender": "Female",
        "phone": "9090909090",
        "email": "karthik@example.com",
        "address": "Madurai, Tamil Nadu",
        "emergency_name": "Ramesh",
        "emergency_phone": "9090908080",
        "blood_group": "O-",
        "allergies": ["Dust"],
        "chronic": [],
    },
    {
        "patient_id": "P006",
        "full_name": "Kanishka",
        "dob": date(1997, 5, 2),
        "gender": "Female",
        "phone": "9555567890",
        "email": "sneha@example.com",
        "address": "Salem, Tamil Nadu",
        "emergency_name": "Raji",
        "emergency_phone": "9555534567",
        "blood_group": "A-",
        "allergies": [],
        "chronic": ["Migraine"],
    },
    {
        "patient_id": "P007",
        "full_name": "Vijay Anand",
        "dob": date(1985, 4, 18),
        "gender": "Male",
        "phone": "8222901100",
        "email": "vijay@example.com",
        "address": "Erode, Tamil Nadu",
        "emergency_name": "Anand",
        "emergency_phone": "8222902200",
        "blood_group": "B-",
        "allergies": ["Peanuts"],
        "chronic": [],
    },
    {
        "patient_id": "P008",
        "full_name": "Aishwarya P",
        "dob": date(1999, 8, 21),
        "gender": "Female",
        "phone": "9123456780",
        "email": "aish@example.com",
        "address": "Trichy, Tamil Nadu",
        "emergency_name": "Padma",
        "emergency_phone": "9123456710",
        "blood_group": "AB-",
        "allergies": ["Sulfa drugs"],
        "chronic": ["Thyroid Disorder"],
    },
    {
        "patient_id": "P009",
        "full_name": "Rahul Dev",
        "dob": date(2003, 12, 30),
        "gender": "Male",
        "phone": "8000000001",
        "email": "rahul@example.com",
        "address": "Mumbai, Maharashtra",
        "emergency_name": "Devraj",
        "emergency_phone": "8000001000",
        "blood_group": "O+",
        "allergies": [],
        "chronic": [],
    },
    {
        "patient_id": "P010",
        "full_name": "Keerthana S",
        "dob": date(1992, 10, 10),
        "gender": "Female",
        "phone": "7339004422",
        "email": "keerthi@example.com",
        "address": "Kochi, Kerala",
        "emergency_name": "Somasundaram",
        "emergency_phone": "7339004433",
        "blood_group": "A+",
        "allergies": ["Latex"],
        "chronic": ["PCOS"],
    },
]

# ---------------------------------------------------------------------
# Insert patients
# ---------------------------------------------------------------------
print("👥 Creating 10 demo patients...")

for p in patients_data:
    patient = Patient(
        patient_id=p["patient_id"],
        full_name=p["full_name"],
        date_of_birth=p["dob"],
        gender=p["gender"],
        phone=p["phone"],
        email=p["email"],
        address=p["address"],
        emergency_contact_name=p["emergency_name"],
        emergency_contact_phone=p["emergency_phone"],
        blood_group=p["blood_group"],
        allergies=p["allergies"],
        chronic_conditions=p["chronic"],
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)

    # Insurance
    ins = InsuranceProfile(
        patient_id=patient.id,
        provider_name="Star Health Insurance",
        policy_number=f"POL{patient.id:05d}",
        coverage_details="Outpatient + Inpatient coverage",
        billing_notes="No pending dues.",
    )
    db.add(ins)

    # No doctor access initially
    db.add(PatientDoctorAccess(
        patient_id=patient.id,
        doctor_username="doc1",
        is_allowed=False
    ))

db.commit()

print("✅ Patients + Insurance + Access seeded successfully!")

db.close()
print("🎉 Done!")
