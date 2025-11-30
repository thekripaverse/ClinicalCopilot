from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np


# Folder to store per-patient face templates
FACE_DB_DIR = Path(__file__).parent / "face_db"
FACE_DB_DIR.mkdir(parents=True, exist_ok=True)

# Use the same Haar cascade you already used
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_CASCADE = cv2.CascadeClassifier(CASCADE_PATH)

if FACE_CASCADE.empty():
    raise RuntimeError(f"Failed to load Haar cascade from {CASCADE_PATH}")


def _face_path(patient_id: str) -> Path:
    """
    Return path to the stored face template for this patient.
    """
    safe_id = patient_id.replace("/", "_").replace("\\", "_")
    return FACE_DB_DIR / f"{safe_id}.npy"


def _extract_face_gray(image_bgr: np.ndarray, size=(100, 100)) -> np.ndarray:
    """
    Detects the largest face in the BGR image, converts to grayscale,
    crops, resizes, and returns a float32 array in [0, 1].
    Raises ValueError if no face found.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = FACE_CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise ValueError("No face detected in image for enrollment/verification.")

    # Take the largest face (by area)
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    face_crop = gray[y : y + h, x : x + w]

    face_resized = cv2.resize(face_crop, size, interpolation=cv2.INTER_AREA)
    face_norm = face_resized.astype("float32") / 255.0
    return face_norm


def enroll_from_image_bytes(patient_id: str, image_bytes: bytes) -> Dict[str, Any]:
    """
    Enrollment:
      - Decodes image bytes
      - Detects face
      - Stores normalized 100x100 grayscale template in face_db/<patient_id>.npy
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode enrollment image.")

    face_template = _extract_face_gray(img, size=(100, 100))
    path = _face_path(patient_id)
    np.save(path, face_template)

    return {
        "patient_id": patient_id,
        "template_path": str(path),
        "shape": face_template.shape,
    }


def verify_from_image_bytes(
    patient_id: str,
    image_bytes: bytes,
    threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    Verification:
      - Loads stored face template for patient
      - Extracts face from current image
      - Computes mean squared difference between the two 100x100 patches.

    Returns:
      {
        "status": "ok" | "no_enrollment" | "decode_error" | "no_face",
        "match": bool,
        "distance": float | None,
        "threshold": float,
      }
    """
    path = _face_path(patient_id)
    if not path.exists():
        return {
            "status": "no_enrollment",
            "match": False,
            "distance": None,
            "threshold": threshold,
        }

    stored_template = np.load(path)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "status": "decode_error",
            "match": False,
            "distance": None,
            "threshold": threshold,
        }

    try:
        current_face = _extract_face_gray(img, size=(100, 100))
    except ValueError:
        return {
            "status": "no_face",
            "match": False,
            "distance": None,
            "threshold": threshold,
        }

    # Mean squared error between normalized templates
    diff = stored_template - current_face
    mse = float(np.mean(diff ** 2))  # 0 = identical, higher = different
    is_match = mse <= threshold

    return {
        "status": "ok",
        "match": bool(is_match),
        "distance": mse,
        "threshold": threshold,
    }
