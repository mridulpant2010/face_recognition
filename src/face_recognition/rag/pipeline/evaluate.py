import json
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..core import config
from .retrieve import FaceRetriever
from ..core.exceptions import ProbeSetNotFoundError

logger = logging.getLogger(__name__)

def compute_average_precision(retrieved_identities: list[str], true_identity: str) -> float:
    if not retrieved_identities:
        return 0.0

    hits = sum_precisions = 0.0
    for k, identity in enumerate(retrieved_identities, start=1):
        if identity == true_identity:
            hits += 1
            sum_precisions += hits / k

    return sum_precisions / hits if hits > 0 else 0.0

def run_evaluation(retriever: FaceRetriever | None = None, top_k: int = config.TOP_K, threshold: float = config.SIMILARITY_THRESHOLD) -> dict:
    logger.info("--- Starting Evaluation Pipeline ---")
    if retriever is None:
        from ..implementations.arcface_embedder import ArcFaceEmbedder
        from ..implementations.faiss_vector_store import FaissVectorStore

        embedder = ArcFaceEmbedder()
        store = FaissVectorStore()
        store.load()
        retriever = FaceRetriever(embedder, store)

    probe_path = config.FAISS_INDEX_DIR / "probe_set.json"
    if not probe_path.exists():
        logger.error("Probe set not found.")
        raise ProbeSetNotFoundError("Probe set not found. Run ingestion first.")

    logger.info(f"Loading probe set from {probe_path}")
    with open(probe_path, "r", encoding="utf-8") as f:
        probe_map = json.load(f)

    rank1_correct = rank5_correct = tar_accepted = total_queried = no_face_count = 0
    all_aps = []

    logger.info("Querying probe set against vector store...")
    for identity, image_paths in tqdm(probe_map.items(), desc="Evaluating"):
        for img_path in image_paths:
            result = retriever.query(img_path, top_k=top_k, threshold=threshold)

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
        "total_queried": total_queried,
        "no_face_detected": no_face_count,
        "rank_1_accuracy": rank1_correct / total_queried if total_queried else 0,
        "rank_5_accuracy": rank5_correct / total_queried if total_queried else 0,
        "mean_average_precision": float(np.mean(all_aps)) if all_aps else 0.0,
        "tar_at_threshold": tar_accepted / total_queried if total_queried else 0,
    }

    logger.info("Evaluation complete. Metric summary:")
    for k, v in metrics.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")

    metrics_path = config.FAISS_INDEX_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")

    return metrics