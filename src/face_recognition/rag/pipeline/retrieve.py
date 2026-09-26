import logging

from ..core import config
from ..interfaces.base_embedder import BaseEmbedder
from ..interfaces.base_vector_store import BaseVectorStore
from ..core.exceptions import ImageReadError, EmptyIndexError

logger = logging.getLogger(__name__)

class FaceRetriever:
    def __init__(self, embedder: BaseEmbedder, store: BaseVectorStore):
        self.embedder = embedder
        self.store = store
        logger.info("FaceRetriever initialized.")

    def query(self, image_path: str, top_k: int = config.TOP_K, threshold: float = config.SIMILARITY_THRESHOLD) -> dict:
        logger.debug(f"Processing query for image: {image_path}")
        result = {
            "query_image": image_path,
            "embedding_extracted": False,
            "predicted_identity": "UNKNOWN",
            "top_match_similarity": 0.0,
            "matches": [],
        }

        try:
            query_emb = self.embedder.embed(image_path)
        except (ImageReadError, Exception) as e:
            logger.warning(f"Failed to embed query image {image_path}: {e}")
            return result

        if query_emb is None:
            logger.debug(f"No face found in query image: {image_path}")
            return result

        result["embedding_extracted"] = True

        try:
            raw_matches = self.store.search(query_emb, top_k=top_k)
        except EmptyIndexError:
            logger.error("Attempted to search an empty index.")
            return result

        filtered_matches = [m for m in raw_matches if m["similarity"] >= threshold]
        result["matches"] = filtered_matches

        if filtered_matches:
            result["top_match_similarity"] = filtered_matches[0]["similarity"]
            result["predicted_identity"] = filtered_matches[0]["identity"]
            logger.debug(f"Found match: {result['predicted_identity']} (Score: {result['top_match_similarity']:.4f})")
        else:
            logger.debug("No matches found above threshold.")

        return result

    def query_batch(self, image_paths: list[str], top_k: int = config.TOP_K, threshold: float = config.SIMILARITY_THRESHOLD) -> list[dict]:
        logger.info(f"Running batch query for {len(image_paths)} images...")
        return [self.query(p, top_k, threshold) for p in image_paths]