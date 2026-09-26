import argparse
import logging
import sys

from .core import config
from .core.exceptions import FaceRAGError
from .pipeline.ingest import run_ingestion
from .pipeline.retrieve import FaceRetriever
from .pipeline.evaluate import run_evaluation

# Configure standard root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_pipeline")

def create_embedder():
    from .implementations.arcface_embedder import ArcFaceEmbedder
    return ArcFaceEmbedder()

def create_store(load_existing: bool = False):
    from .implementations.faiss_vector_store import FaissVectorStore
    store = FaissVectorStore()
    if load_existing:
        store.load()
    return store

def handle_query(query_image: str) -> None:
    logger.info(f"Running single query for {query_image}")
    embedder = create_embedder()
    store = create_store(load_existing=True)
    retriever = FaceRetriever(embedder, store)

    result = retriever.query(query_image)
    logger.info(f"Predicted ID: {result['predicted_identity']} (Score: {result['top_match_similarity']:.4f})")
    
    if result["matches"]:
        logger.info("Top Matches:")
        for m in result["matches"]:
            logger.info(f"  [{m['rank']}] {m['identity']} ({m['similarity']:.4f})")

def main():
    parser = argparse.ArgumentParser(description="Face Recognition RAG Pipeline")
    parser.add_argument("--ingest-only", action="store_true", help="Run ingestion phase")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation phase")
    parser.add_argument("--query", type=str, help="Path to query image")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.query:
            handle_query(args.query)
        elif args.eval_only:
            run_evaluation(retriever=FaceRetriever(create_embedder(), create_store(True)))
        elif args.ingest_only:
            run_ingestion(embedder=create_embedder(), store=create_store(False))
        else:
            logger.info("Running full pipeline (Ingest -> Evaluate)")
            embedder = create_embedder()
            run_ingestion(embedder=embedder, store=create_store(False))
            run_evaluation(retriever=FaceRetriever(embedder, create_store(True)))
            
    except FaceRAGError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()