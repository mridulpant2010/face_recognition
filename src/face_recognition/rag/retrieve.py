"""
Retrieval Module
================
Phase 2 of the RAG pipeline.

Given a query face image, this module:
    1. Embeds the query face with ArcFace (same model used during ingestion).
    2. Searches the FAISS vector store for the Top-K nearest neighbours.
    3. Applies a similarity threshold to filter out false matches.
    4. Returns the retrieved identity matches with metadata and scores.
"""

import numpy as np

from . import config
from .embedder import ArcFaceEmbedder
from .vector_store import FaissVectorStore


class FaceRetriever:
    """
    Takes a query image and retrieves the closest identities from the
    FAISS vector store.

    Usage:
        retriever = FaceRetriever()
        results = retriever.query("path/to/unknown_face.jpg")
        for r in results:
            print(r["identity"], r["similarity"])
    """

    def __init__(self, embedder: ArcFaceEmbedder | None = None):
        """
        Load the embedder and the saved FAISS index from disk.
        """
        self.embedder = embedder or ArcFaceEmbedder()
        self.store = FaissVectorStore()
        self.store.load()  # Loads the index saved during ingestion
        print(f"[Retriever] Ready. Database contains {self.store.total_vectors} face vectors.")

    def query(
        self,
        image_path: str,
        top_k: int = config.TOP_K,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> dict:
        """
        Run a face verification/identification query.

        Args:
            image_path: Path to the query face image.
            top_k:      Number of nearest neighbours to retrieve.
            threshold:  Minimum cosine similarity for a positive match.

        Returns:
            A dict containing:
            {
                "query_image": "path/to/query.jpg",
                "embedding_extracted": True/False,
                "predicted_identity": "n000123" or "UNKNOWN",
                "top_match_similarity": 0.87,
                "matches": [
                    {"identity": "n000123", "image_path": "...", "similarity": 0.87, "rank": 1},
                    {"identity": "n000123", "image_path": "...", "similarity": 0.82, "rank": 2},
                    ...
                ]
            }
        """
        result = {
            "query_image": image_path,
            "embedding_extracted": False,
            "predicted_identity": "UNKNOWN",
            "top_match_similarity": 0.0,
            "matches": [],
        }

        # Step 1: Embed the query image
        query_emb = self.embedder.embed(image_path)
        if query_emb is None:
            return result

        result["embedding_extracted"] = True

        # Step 2: Search the FAISS index
        raw_matches = self.store.search(query_emb, top_k=top_k)

        # Step 3: Filter by threshold
        filtered_matches = [m for m in raw_matches if m["similarity"] >= threshold]
        result["matches"] = filtered_matches

        # Step 4: Determine the predicted identity
        if filtered_matches:
            result["top_match_similarity"] = filtered_matches[0]["similarity"]
            result["predicted_identity"] = filtered_matches[0]["identity"]

        return result

    def query_batch(
        self,
        image_paths: list[str],
        top_k: int = config.TOP_K,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> list[dict]:
        """
        Run queries for multiple images. Returns a list of result dicts.
        """
        return [self.query(p, top_k, threshold) for p in image_paths]


def run_retrieval_demo(query_image: str) -> None:
    """
    Quick demo: query a single image against the stored database.
    """
    print("=" * 60)
    print("FACE RAG — RETRIEVAL DEMO")
    print("=" * 60)

    retriever = FaceRetriever()
    result = retriever.query(query_image)

    print(f"\n  Query Image      : {result['query_image']}")
    print(f"  Face Detected    : {result['embedding_extracted']}")
    print(f"  Predicted ID     : {result['predicted_identity']}")
    print(f"  Top Match Score  : {result['top_match_similarity']:.4f}")
    print(f"  Matches Found    : {len(result['matches'])}")

    if result["matches"]:
        print("\n  Top-K Matches:")
        print(f"  {'Rank':<6} {'Identity':<12} {'Similarity':<12} {'Image Path'}")
        print("  " + "-" * 70)
        for m in result["matches"]:
            print(f"  {m['rank']:<6} {m['identity']:<12} {m['similarity']:<12.4f} {m['image_path']}")

    print("=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m face_recognition.rag.retrieve <path_to_query_image>")
    else:
        run_retrieval_demo(sys.argv[1])
