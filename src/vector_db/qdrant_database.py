from qdrant_client import QdrantClient
from qdrant_client import models
from typing import List, Dict, Any

class QdrantDBClient:
    def __init__(self, host: str, api_key: str, collection_name: str):
        self.client = QdrantClient(host=host, api_key=api_key, timeout=120.0)
        self.collection_name = collection_name

    def ensure_collection(self, vector_size: int):
        collections = self.client.get_collections().collections
        existing = any(c.name == self.collection_name for c in collections)

        if not existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size, 
                    distance=models.Distance.COSINE, 
                    on_disk=True,
                    hnsw_config=models.HnswConfigDiff(
                        m=64,
                        ef_construct=512,
                        on_disk=True,
                    )
                )
            )

    def upsert_vectors(self,  points: List[Dict[str, Any]]):
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search_vectors(
        self, 
        query_vector: list, 
        top_k: int, 
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        raw_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True  # ensure metadata is returned
        )

        results = []
        for r in raw_results:
            score = getattr(r, "score", None)
            if score is not None and score < score_threshold:
                continue
            results.append({
                "id": r.id,
                "score": score,
                "metadata": r.payload,
                "text": r.payload.get("text")  # assuming you stored text in payload
            })

        return results
    
    def delete_collection(self):
        self.client.delete_collection(collection_name=self.collection_name)