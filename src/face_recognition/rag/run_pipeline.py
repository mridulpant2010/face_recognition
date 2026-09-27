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


# ---- Unimodal Factories (Image-only, ArcFace) ----

def create_embedder():
    from .implementations.arcface_embedder import ArcFaceEmbedder
    return ArcFaceEmbedder()

def create_store(load_existing: bool = False, dim: int = config.EMBEDDING_DIM):
    from .implementations.faiss_vector_store import FaissVectorStore
    store = FaissVectorStore(dim=dim)
    if load_existing:
        store.load()
    return store


# ---- Multimodal Factories (Image + Text) ----

def create_multimodal_embedder(fusion: str = config.FUSION_STRATEGY):
    from .implementations.arcface_embedder import ArcFaceEmbedder
    from .implementations.text_encoder import SentenceTransformerEncoder
    from .implementations.multimodal_embedder import MultimodalEmbedder

    image_embedder = ArcFaceEmbedder()
    text_encoder = SentenceTransformerEncoder(model_name=config.TEXT_ENCODER_MODEL)

    return MultimodalEmbedder(
        image_embedder=image_embedder,
        text_encoder=text_encoder,
        fusion=fusion,
        image_weight=config.IMAGE_WEIGHT,
    )

def create_multimodal_store(load_existing: bool = False, fusion: str = config.FUSION_STRATEGY):
    from .implementations.faiss_vector_store import FaissVectorStore
    from .implementations.text_encoder import SentenceTransformerEncoder

    # Determine output dimension based on fusion strategy
    if fusion == "concat":
        temp_encoder = SentenceTransformerEncoder(model_name=config.TEXT_ENCODER_MODEL)
        dim = config.EMBEDDING_DIM + temp_encoder.embedding_dim
    else:
        dim = config.EMBEDDING_DIM

    store = FaissVectorStore(dim=dim)
    
    # Store in a subfolder specific to the fusion strategy to prevent overwriting
    store_dir = config.MULTIMODAL_INDEX_DIR / fusion
    
    if load_existing:
        store.load(store_dir)
    return store


# ---- Handlers ----

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

def handle_multimodal_query(query_image: str, query_text: str | None = None) -> None:
    from .pipeline.multimodal_evaluate import MultimodalRetriever

    logger.info(f"Running multimodal query for {query_image}")
    mm_embedder = create_multimodal_embedder()
    store = create_multimodal_store(load_existing=True)
    retriever = MultimodalRetriever(mm_embedder, store)

    result = retriever.query(query_image, text=query_text)
    logger.info(f"Predicted ID: {result['predicted_identity']} (Score: {result['top_match_similarity']:.4f})")

    if result["matches"]:
        logger.info("Top Matches:")
        for m in result["matches"]:
            logger.info(f"  [{m['rank']}] {m['identity']} ({m['similarity']:.4f})")

def handle_unimodal_pipeline(ingest_only: bool = False, eval_only: bool = False) -> None:
    embedder = create_embedder()
    if eval_only:
        run_evaluation(retriever=FaceRetriever(embedder, create_store(True)))
    elif ingest_only:
        run_ingestion(embedder=embedder, store=create_store(False))
    else:
        logger.info("Running full unimodal pipeline (Ingest -> Evaluate)")
        run_ingestion(embedder=embedder, store=create_store(False))
        run_evaluation(retriever=FaceRetriever(embedder, create_store(True)))

def handle_multimodal_pipeline(
    ingest_only: bool = False,
    eval_only: bool = False,
    fusion: str = config.FUSION_STRATEGY,
) -> None:
    from .pipeline.multimodal_ingest import run_multimodal_ingestion
    from .pipeline.multimodal_evaluate import MultimodalRetriever, run_multimodal_evaluation

    mm_embedder = create_multimodal_embedder(fusion=fusion)
    
    # Store in a subfolder specific to the fusion strategy to prevent overwriting
    store_dir = config.MULTIMODAL_INDEX_DIR / fusion

    if eval_only:
        store = create_multimodal_store(load_existing=True, fusion=fusion)
        retriever = MultimodalRetriever(mm_embedder, store)
        run_multimodal_evaluation(retriever, store_dir)
    elif ingest_only:
        store = create_multimodal_store(load_existing=False, fusion=fusion)
        run_multimodal_ingestion(mm_embedder, store, store_dir)
    else:
        logger.info(f"Running full multimodal pipeline (fusion: {fusion})")
        store = create_multimodal_store(load_existing=False, fusion=fusion)
        run_multimodal_ingestion(mm_embedder, store, store_dir)

        eval_store = create_multimodal_store(load_existing=True, fusion=fusion)
        retriever = MultimodalRetriever(mm_embedder, eval_store)
        run_multimodal_evaluation(retriever, store_dir)


def main():
    parser = argparse.ArgumentParser(description="Face Recognition RAG Pipeline")
    parser.add_argument("--ingest-only", action="store_true", help="Run ingestion phase only")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation phase only")
    parser.add_argument("--query", type=str, help="Path to query image")
    parser.add_argument("--query-text", type=str, default=None, help="Text description for multimodal query")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # Multimodal arguments
    parser.add_argument("--multimodal", action="store_true", help="Use multimodal (Image+Text) pipeline")
    parser.add_argument(
        "--fusion", type=str, default=config.FUSION_STRATEGY,
        choices=["concat", "add", "weighted_add"],
        help="Fusion strategy for multimodal embeddings",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        if args.query:
            if args.multimodal:
                handle_multimodal_query(args.query, args.query_text)
            else:
                handle_query(args.query)
        elif args.multimodal:
            handle_multimodal_pipeline(args.ingest_only, args.eval_only, args.fusion)
        else:
            handle_unimodal_pipeline(args.ingest_only, args.eval_only)

    except FaceRAGError as e:
        logger.error(f"Pipeline error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted.")
        sys.exit(130)

if __name__ == "__main__":
    main()