"""
Evaluation Module
=================
Phase 3 of the RAG pipeline.

Loads the saved probe set (the 20% held-out images per identity),
queries each probe image against the FAISS gallery, and computes
standard retrieval metrics:

    1. Rank-1 Accuracy  : Is the #1 retrieved chunk from the correct identity?
    2. Rank-5 Accuracy  : Is the correct identity in the top 5 retrieved chunks?
    3. Mean Average Precision (mAP) : How well are ALL correct chunks ranked?
    4. True Accept Rate (TAR) @ threshold : What fraction of genuine queries
       have a top-match similarity above the threshold?
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import config
from .embedder import ArcFaceEmbedder
from .retrieve import FaceRetriever


def compute_average_precision(retrieved_identities: list[str], true_identity: str) -> float:
    """
    Compute the Average Precision (AP) for a single query.

    AP = (1 / number_of_relevant_docs) * Σ (Precision@k * rel(k))

    where rel(k) = 1 if the k-th retrieved item is relevant, else 0.

    Args:
        retrieved_identities: List of identity strings from the ranked results.
        true_identity:        The ground-truth identity of the query.

    Returns:
        Average Precision (float between 0.0 and 1.0).
    """
    if not retrieved_identities:
        return 0.0

    hits = 0
    sum_precisions = 0.0

    for k, identity in enumerate(retrieved_identities, start=1):
        if identity == true_identity:
            hits += 1
            precision_at_k = hits / k
            sum_precisions += precision_at_k

    if hits == 0:
        return 0.0

    return sum_precisions / hits


def run_evaluation(
    top_k: int = config.TOP_K,
    threshold: float = config.SIMILARITY_THRESHOLD,
) -> dict:
    """
    Full evaluation pipeline:
        1. Load the probe set from disk.
        2. Query each probe image against the FAISS gallery.
        3. Compute Rank-1, Rank-5, mAP, and TAR@threshold.

    Returns:
        A dict containing all computed metrics.
    """
    print("=" * 60)
    print("FACE RAG — EVALUATION PIPELINE")
    print("=" * 60)

    # --- Step 1: Load the probe set ---
    probe_path = config.FAISS_INDEX_DIR / "probe_set.json"
    if not probe_path.exists():
        print(f"  ERROR: Probe set not found at {probe_path}.")
        print("  Run the ingestion pipeline first (ingest.py).")
        return {}

    with open(probe_path, "r") as f:
        probe_map = json.load(f)

    total_probes = sum(len(imgs) for imgs in probe_map.values())
    print(f"\n[Step 1] Loaded probe set: {total_probes} images across {len(probe_map)} identities.")

    # --- Step 2: Initialise the retriever ---
    print(f"\n[Step 2] Initialising retriever...")
    retriever = FaceRetriever()

    # --- Step 3: Query every probe image ---
    print(f"\n[Step 3] Running retrieval queries...")

    rank1_correct = 0
    rank5_correct = 0
    tar_accepted = 0
    all_aps = []
    total_queried = 0
    no_face_count = 0

    for identity, image_paths in tqdm(probe_map.items(), desc="Evaluating"):
        for img_path in image_paths:
            result = retriever.query(img_path, top_k=top_k, threshold=threshold)

            if not result["embedding_extracted"]:
                no_face_count += 1
                continue

            total_queried += 1
            retrieved_ids = [m["identity"] for m in result["matches"]]

            # --- Rank-1 Accuracy ---
            if retrieved_ids and retrieved_ids[0] == identity:
                rank1_correct += 1

            # --- Rank-5 Accuracy ---
            if identity in retrieved_ids[:5]:
                rank5_correct += 1

            # --- TAR @ threshold ---
            if result["top_match_similarity"] >= threshold and result["predicted_identity"] == identity:
                tar_accepted += 1

            # --- Average Precision ---
            # For mAP, we look at all retrieved results (not just above threshold)
            all_retrieved = [m["identity"] for m in retriever.store.search(
                retriever.embedder.embed(img_path), top_k=top_k
            )]
            ap = compute_average_precision(all_retrieved, identity)
            all_aps.append(ap)

    # --- Step 4: Compute final metrics ---
    metrics = {}
    if total_queried > 0:
        metrics = {
            "total_probe_images": total_probes,
            "total_queried": total_queried,
            "no_face_detected": no_face_count,
            "rank_1_accuracy": rank1_correct / total_queried,
            "rank_5_accuracy": rank5_correct / total_queried,
            "mean_average_precision": float(np.mean(all_aps)) if all_aps else 0.0,
            "tar_at_threshold": tar_accepted / total_queried,
            "threshold_used": threshold,
        }

    # --- Print Results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Probe Images       : {total_probes}")
    print(f"  Successfully Queried     : {total_queried}")
    print(f"  No Face Detected (skip)  : {no_face_count}")
    print(f"  -----------------------------------------")
    print(f"  Rank-1 Accuracy          : {metrics.get('rank_1_accuracy', 0):.4f}  ({rank1_correct}/{total_queried})")
    print(f"  Rank-5 Accuracy          : {metrics.get('rank_5_accuracy', 0):.4f}  ({rank5_correct}/{total_queried})")
    print(f"  Mean Average Precision   : {metrics.get('mean_average_precision', 0):.4f}")
    print(f"  TAR @ threshold={threshold:.2f}    : {metrics.get('tar_at_threshold', 0):.4f}  ({tar_accepted}/{total_queried})")
    print("=" * 60)

    # --- Save metrics to disk ---
    metrics_path = config.FAISS_INDEX_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved → {metrics_path}")

    return metrics


if __name__ == "__main__":
    run_evaluation()
