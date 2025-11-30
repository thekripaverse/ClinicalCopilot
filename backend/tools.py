from typing import Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from datetime import datetime
import speech_recognition as sr

QDRANT_PATH = Path(__file__).parent / "qdrant_local"
QDRANT_PATH.mkdir(exist_ok=True)

_qdrant_client = QdrantClient(
    path=str(QDRANT_PATH),
)

def get_qdrant_client() -> QdrantClient:
    """
    Reuse the single embedded Qdrant client instance,
    so we don't create multiple clients on the same local folder.
    """
    return _qdrant_client

_embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

GUIDELINE_COLLECTION = "clinical_guidelines"
def _ensure_guideline_collection():
    collections = _qdrant_client.get_collections().collections
    existing = {c.name for c in collections}

    if GUIDELINE_COLLECTION not in existing:
        _qdrant_client.recreate_collection(
            collection_name=GUIDELINE_COLLECTION,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )


def rag_query_tool(query: str, top_k: int = 3):
    try:
        _ensure_guideline_collection()
        vec = _embedder.encode(query).tolist()

        results = _qdrant_client.search(
            collection_name=GUIDELINE_COLLECTION,
            query_vector=vec,
            limit=top_k,
        )

        output = []
        for r in results:
            output.append({
                "text": r.payload.get("text", ""),
                "source": r.payload.get("source", ""),
                "score": r.score
            })
        return output

    except Exception as e:
        print("❌ RAG error:", e)
        return []


def tool_order_test(test_name: str) -> Dict[str, Any]:
    """
    Mock test ordering tool.
    In real deployment, this would call a lab system via MCP/FHIR.
    """
    return {
        "action": "order_test",
        "test_name": test_name,
        "status": "ordered_mock",
        "order_id": f"LAB-MOCK-{test_name.upper()}",
    }

EMR_STORE_PATH = Path(__file__).parent / "emr_store.json"
def tool_update_emr(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock EMR update tool.
    Now also PERSISTS records to a local JSON file (emr_store.json)
    so you can prove that data is stored.
    """
    record = {
        "emr_record_id": f"EMR-{int(datetime.utcnow().timestamp())}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        **payload,
    }

    # Load existing EMR records if file exists
    records: list[Dict[str, Any]] = []
    if EMR_STORE_PATH.exists():
        try:
            with EMR_STORE_PATH.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except Exception:
            records = []

    # Append new record
    records.append(record)

    # Save back to file
    with EMR_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    return {
        "action": "update_emr",
        "status": "success_mock",
        "emr_record_id": record["emr_record_id"],
    }

PHARMACY_STORE_PATH = Path(__file__).parent / "pharmacy_orders.json"

def tool_send_to_pharmacy(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Mock Pharmacy integration tool.

    In a real system this would call an e-prescription / pharmacy API.
    Here we persist an order into pharmacy_orders.json so you can show
    the full pipeline: Doctor -> EMR -> Pharmacy.
    """
    order = {
        "order_id": f"RX-{int(datetime.utcnow().timestamp())}",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "status": "queued_demo",   # e.g. 'queued', 'sent', 'dispensed'
        **payload,
    }

    # Load existing orders if file exists
    orders: list[Dict[str, Any]] = []
    if PHARMACY_STORE_PATH.exists():
        try:
            with PHARMACY_STORE_PATH.open("r", encoding="utf-8") as f:
                orders = json.load(f)
        except Exception:
            orders = []

    # Append new order
    orders.append(order)

    # Save back to file
    with PHARMACY_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(orders, f, indent=2, ensure_ascii=False)

    return order

def tool_transcribe_voice(path: str) -> str:
    recognizer = sr.Recognizer()

    try:
        with sr.AudioFile(path) as source:
            audio_data = recognizer.record(source)

        # Uses Google Web Speech API (no key needed for basic use)
        text = recognizer.recognize_google(audio_data, language="en-US")
        print("🗣️ Google STT transcription:", text)
        return text.strip() or "Transcription empty (no speech detected)."

    except sr.UnknownValueError:
        # Audio was heard but not understandable
        print("❌ Google STT could not understand audio.")
        return "STT could not understand the audio clearly."
    except sr.RequestError as e:
        # Problem reaching Google STT service
        print("🔥 Google STT request error:", repr(e))
        return "STT request failed due to a network or service error."
    except Exception as e:
        print("🔥 General STT backend error:", repr(e))
        return "Transcription failed due to an internal STT error."