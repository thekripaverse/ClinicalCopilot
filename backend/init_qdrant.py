import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "clinical_guidelines"


def init_qdrant():
    client = QdrantClient(url=QDRANT_URL)

    # Create / reset collection
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=rest.VectorParams(
            size=384,  # all-MiniLM-L6-v2 embedding size
            distance=rest.Distance.COSINE,
        ),
    )

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    data_path = Path("guidelines/tests_guidelines.json")
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    ids = []
    vectors = []
    payloads = []

    for item in data:
        text = item["text"]
        embedding = model.encode(text)
        ids.append(item["id"])
        vectors.append(embedding)
        payloads.append(
            {
                "title": item["title"],
                "text": item["text"],
                "tags": item["tags"],
            }
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=rest.Batch(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        ),
    )

    print(f"✅ Qdrant initialized with {len(ids)} guideline entries.")


if __name__ == "__main__":
    init_qdrant()
