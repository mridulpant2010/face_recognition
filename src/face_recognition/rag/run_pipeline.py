"""
RAG Pipeline Runner
===================
Main entry point that orchestrates the full pipeline:

    Step 1: INGEST  — Scan VGGFace2, embed gallery images, populate FAISS.
    Step 2: RETRIEVE — (Optional) Run a single-image demo query.
    Step 3: EVALUATE — Run all probe images through the pipeline and compute metrics.

Usage:
    # Run the full pipeline (ingest + evaluate):
    python -m face_recognition.rag.run_pipeline

    # Run only ingestion:
    python -m face_recognition.rag.run_pipeline --ingest-only

    # Run only evaluation (if you already ingested):
    python -m face_recognition.rag.run_pipeline --eval-only

    # Run a single query demo:
    python -m face_recognition.rag.run_pipeline --query path/to/face.jpg
"""

import argparse

from . import config
from .ingest import run_ingestion
from .retrieve import run_retrieval_demo
from .evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition RAG Pipeline — ArcFace + FAISS"
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run the ingestion phase (embed + store gallery images)."
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation (requires a previously ingested FAISS index)."
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Path to a single query image for a retrieval demo."
    )
    args = parser.parse_args()

    if args.query:
        # Just run a single retrieval demo
        run_retrieval_demo(args.query)
        return

    if args.eval_only:
        run_evaluation()
        return

    if args.ingest_only:
        run_ingestion()
        return

    # Default: run the full pipeline
    print("\n" + "#" * 60)
    print("#  FACE RECOGNITION RAG PIPELINE")
    print(f"#  Dataset    : {config.VGGFACE2_ROOT}")
    print(f"#  Model      : {config.ARCFACE_MODEL_NAME}")
    print(f"#  Gallery %  : {config.GALLERY_RATIO:.0%}")
    print(f"#  Threshold  : {config.SIMILARITY_THRESHOLD}")
    print(f"#  Top-K      : {config.TOP_K}")
    print("#" * 60 + "\n")

    # Phase 1: Ingest
    run_ingestion()

    # Phase 3: Evaluate
    print("\n")
    run_evaluation()


if __name__ == "__main__":
    main()
