import json
import logging
import numpy as np
from tqdm import tqdm

from ..core import config
from ..implementations.multimodal_embedder import MultimodalEmbedder
from ..interfaces.base_vector_store import BaseVectorStore
from ..core.exceptions import ProbeSetNotFoundError, EmptyIndexError

logger = logging.getLogger(__name__)


class MultimodalRetriever:
    def __init__(self, mm_embedder: MultimodalEmbedder, store: BaseVectorStore):
        self.mm_embedder = mm_embedder
        self.store = store
        logger.info("MultimodalRetriever initialized.")

    def query(
        self,
        image_path: str,
        text: str | None = None,
        top_k: int = config.TOP_K,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> dict:
        result = {
            "query_image": image_path,
            "query_text": text,
            "embedding_extracted": False,
            "predicted_identity": "UNKNOWN",
            "top_match_similarity": 0.0,
            "matches": [],
        }

        try:
            if text:
                query_emb = self.mm_embedder.embed(image_path, text)
            else:
                query_emb = self.mm_embedder.embed_image_only(image_path)
        except Exception as e:
            logger.warning(f"Failed to embed query: {e}")
            return result

        if query_emb is None:
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

        return result

    def query_text_only(
        self,
        text: str,
        top_k: int = config.TOP_K,
        threshold: float = config.SIMILARITY_THRESHOLD,
    ) -> dict:
        """Search the database using only a text description (no image)."""
        result = {
            "query_text": text,
            "embedding_extracted": True,
            "predicted_identity": "UNKNOWN",
            "top_match_similarity": 0.0,
            "matches": [],
        }

        query_emb = self.mm_embedder.embed_text_only(text)

        try:
            raw_matches = self.store.search(query_emb, top_k=top_k)
        except EmptyIndexError:
            return result

        filtered_matches = [m for m in raw_matches if m["similarity"] >= threshold]
        result["matches"] = filtered_matches

        if filtered_matches:
            result["top_match_similarity"] = filtered_matches[0]["similarity"]
            result["predicted_identity"] = filtered_matches[0]["identity"]

        return result


def compute_average_precision(retrieved_identities: list[str], true_identity: str) -> float:
    if not retrieved_identities:
        return 0.0

    hits = sum_precisions = 0.0
    for k, identity in enumerate(retrieved_identities, start=1):
        if identity == true_identity:
            hits += 1
            sum_precisions += hits / k

    return sum_precisions / hits if hits > 0 else 0.0


def run_multimodal_evaluation(
    retriever: MultimodalRetriever,
    store_dir: Path,
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> dict:
    logger.info(f"--- Starting Multimodal Evaluation (Loading from {store_dir}) ---")

    probe_path = store_dir / "probe_set.json"
    if not probe_path.exists():
        raise ProbeSetNotFoundError("Multimodal probe set not found. Run multimodal ingestion first.")

    with open(probe_path, "r", encoding="utf-8") as f:
        probe_map = json.load(f)

    rank1_correct = rank5_correct = tar_accepted = total_queried = no_face_count = 0
    all_aps = []

    logger.info("Querying probe set...")
    for identity, entries in tqdm(probe_map.items(), desc="Evaluating"):
        for entry in entries:
            img_path = entry["image_path"]
            caption = entry.get("caption", None)

            result = retriever.query(img_path, text=caption, top_k=top_k, threshold=threshold)

            if not result["embedding_extracted"]:
                no_face_count += 1
                continue

            total_queried += 1
            retrieved_ids = [m["identity"] for m in result["matches"]]

            if retrieved_ids and retrieved_ids[0] == identity:
                rank1_correct += 1
            if identity in retrieved_ids[:5]:
                rank5_correct += 1
            if result["top_match_similarity"] >= threshold and result["predicted_identity"] == identity:
                tar_accepted += 1

            all_aps.append(compute_average_precision(retrieved_ids, identity))

    metrics = {
        "mode": "multimodal",
        "fusion": retriever.mm_embedder._fusion,
        "total_queried": total_queried,
        "no_face_detected": no_face_count,
        "rank_1_accuracy": rank1_correct / total_queried if total_queried else 0,
        "rank_5_accuracy": rank5_correct / total_queried if total_queried else 0,
        "mean_average_precision": float(np.mean(all_aps)) if all_aps else 0.0,
        "tar_at_threshold": tar_accepted / total_queried if total_queried else 0,
    }

    logger.info("Multimodal evaluation complete:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    metrics_path = store_dir / "evaluation_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    return metrics
