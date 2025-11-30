from pathlib import Path
from datetime import datetime, date

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Integer,
    Date,
    DateTime,
    Boolean,
    Text,
    ForeignKey,
)
from sqlalchemy.dialects.sqlite import JSON as SQLiteJSON
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# ---------- DB setup ----------

DB_PATH = Path(__file__).parent / "ehr.db"
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},  # needed for SQLite + FastAPI
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# ---------- Core EHR models ----------

class Patient(Base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(String, unique=True, index=True, nullable=False)

    full_name = Column(String, nullable=False)
    date_of_birth = Column(Date, nullable=True)
    gender = Column(String, nullable=True)

    phone = Column(String, nullable=True)
    email = Column(String, nullable=True)
    address = Column(Text, nullable=True)

    emergency_contact_name = Column(String, nullable=True)
    emergency_contact_phone = Column(String, nullable=True)

    # NEW health context
    blood_group = Column(String, nullable=True)           # e.g. "O+", "A-"
    allergies = Column(SQLiteJSON, nullable=True)         # list[str]
    chronic_conditions = Column(SQLiteJSON, nullable=True)  # list[str], e.g. ["Hypertension", "Type 2 DM"]

    # Relationships
    insurance_profile = relationship("InsuranceProfile", back_populates="patient", uselist=False)
    encounters = relationship("Encounter", back_populates="patient")
    lab_results = relationship("LabResult", back_populates="patient")
    radiology_reports = relationship("RadiologyReport", back_populates="patient")
    pharmacy_orders = relationship("PharmacyOrder", back_populates="patient")

class InsuranceProfile(Base):
    """
    Administrative / billing information.
    """
    __tablename__ = "insurance_profiles"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)

    provider_name = Column(String, nullable=True)
    policy_number = Column(String, nullable=True)
    coverage_details = Column(Text, nullable=True)
    billing_notes = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="insurance_profile")

class PatientDoctorAccess(Base):
    __tablename__ = "patient_doctor_access"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_username = Column(String, nullable=False)
    is_allowed = Column(Boolean, default=False)

    patient = relationship("Patient")

class Encounter(Base):
    __tablename__ = "encounters"

    id = Column(Integer, primary_key=True, index=True)
    encounter_id = Column(String, unique=True, index=True, nullable=False)

    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    doctor_username = Column(String, nullable=True)

    # NEW - high level consultation info
    chief_complaint = Column(String, nullable=True)   # e.g. "Fever and cough for 3 days"
    visit_type = Column(String, nullable=True)        # e.g. "OPD", "Follow-up"

    # Clinical data
    note_summary = Column(Text, nullable=True)       # progress note / summary
    problems = Column(SQLiteJSON, nullable=True)     # list[str] problems / diagnoses
    medications = Column(SQLiteJSON, nullable=True)  # list[str] meds
    vitals = Column(SQLiteJSON, nullable=True)       # dict: pulse, bp, temp, etc.
    past_medical_history = Column(SQLiteJSON, nullable=True)  # list[str]

    symptoms = Column(SQLiteJSON, nullable=True)          # list[str]
    suggested_tests = Column(SQLiteJSON, nullable=True)   # list[str]

    prescription = Column(Text, nullable=True)
    approved_by_doctor = Column(Boolean, default=False)

    patient = relationship("Patient", back_populates="encounters")
    lab_results = relationship("LabResult", back_populates="encounter")
    radiology_reports = relationship("RadiologyReport", back_populates="encounter")


class LabResult(Base):
    """
    Laboratory test results.
    """
    __tablename__ = "lab_results"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=True)

    test_name = Column(String, nullable=False)
    result_value = Column(String, nullable=True)
    unit = Column(String, nullable=True)
    reference_range = Column(String, nullable=True)
    status = Column(String, nullable=True)  # ordered, completed, flagged
    report_text = Column(Text, nullable=True)

    patient = relationship("Patient", back_populates="lab_results")
    encounter = relationship("Encounter", back_populates="lab_results")


class RadiologyReport(Base):
    """
    Radiology reports (X-ray, CT, MRI).
    """
    __tablename__ = "radiology_reports"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=True)

    modality = Column(String, nullable=True)     # e.g. X-ray, CT, MRI
    body_part = Column(String, nullable=True)    # e.g. Chest
    impression = Column(Text, nullable=True)     # short impression
    report_text = Column(Text, nullable=True)    # full report

    patient = relationship("Patient", back_populates="radiology_reports")
    encounter = relationship("Encounter", back_populates="radiology_reports")


class PharmacyOrder(Base):
    """
    Orders sent to pharmacy.
    """
    __tablename__ = "pharmacy_orders"

    id = Column(Integer, primary_key=True, index=True)
    order_id = Column(String, unique=True, index=True, nullable=False)

    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    encounter_id = Column(Integer, ForeignKey("encounters.id"), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    prescription = Column(Text, nullable=True)
    status = Column(String, default="pending")

    patient = relationship("Patient", back_populates="pharmacy_orders")

def init_db():
    """Create tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_or_create_patient(db, external_patient_id: str) -> Patient:
    patient = db.query(Patient).filter(Patient.patient_id == external_patient_id).first()
    if patient:
        return patient

    patient = Patient(
        patient_id=external_patient_id,
        full_name=external_patient_id,  # placeholder (you can update later)
    )
    db.add(patient)
    db.commit()
    db.refresh(patient)
    return patient
