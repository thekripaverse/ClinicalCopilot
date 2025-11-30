import json
from pathlib import Path
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ✅ Use the SAME path as tools.py
QDRANT_PATH = Path(__file__).parent / "qdrant_local"
COLLECTION_NAME = "clinical_guidelines"
EMBED_DIM = 512  # must match whatever your embed_texts() uses

# --- Very simple demo embedding (replace with your real model if you want) ---
def dummy_embed(texts: List[str]) -> np.ndarray:
    """
    Minimal fallback embedding: turn text into a fixed-size numeric vector.
    This is ONLY for demo. For your project, you probably already
    have a proper embed_texts() in tools.py.
    """
    vecs = []
    for t in texts:
        # hash-based toy embedding just so we can store vectors
        arr = np.zeros(EMBED_DIM, dtype=np.float32)
        for i, ch in enumerate(t.encode("utf-8")):
            arr[i % EMBED_DIM] += float(ch)
        vecs.append(arr / (np.linalg.norm(arr) + 1e-6))
    return np.stack(vecs, axis=0)


def main():
    print(f"Using Qdrant at: {QDRANT_PATH}")
    client = QdrantClient(path=str(QDRANT_PATH))

    # 1) Create collection if it does not exist
    if COLLECTION_NAME not in [c.name for c in client.get_collections().collections]:
        print(f"Creating collection: {COLLECTION_NAME}")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print(f"Collection {COLLECTION_NAME} already exists.")

    # 2) Example guideline chunks (you can later replace with PDF/text loader)
    guideline_chunks = [
        "For fever and sore throat, consider CBC, CRP, and throat swab if symptoms persist.",
        "For chest pain with cardiovascular risk factors, order ECG 12-lead, Troponin, and Chest X-ray.",
        "For suspected diabetes, order Fasting blood sugar, HbA1c, and Lipid profile.",
        "For unexplained weight loss, evaluate with CBC, LFT, RFT, and Thyroid profile.",
        "For chronic cough, consider Chest X-ray, Spirometry, and sputum examination."
    ]

    print(f"Embedding {len(guideline_chunks)} guideline chunks...")
    vectors = dummy_embed(guideline_chunks)

    points = []
    for idx, (text, vec) in enumerate(zip(guideline_chunks, vectors), start=1):
        points.append(
            PointStruct(
                id=idx,
                vector=vec.tolist(),
                payload={"text": text}
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    count = client.count(collection_name=COLLECTION_NAME, exact=True).count
    print(f"✅ Upserted {len(points)} points. Collection now has {count} points.")


if __name__ == "__main__":
    main()
