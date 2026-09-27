import json
import logging
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..core import config
from ..interfaces.base_embedder import BaseEmbedder
from ..interfaces.base_vector_store import BaseVectorStore
from ..core.exceptions import DatasetNotFoundError, ImageReadError

logger = logging.getLogger(__name__)

def discover_dataset(dataset_root: Path) -> dict[str, list[Path]]:
    logger.info(f"Discovering dataset in {dataset_root}...")
    if not dataset_root.exists() or not dataset_root.is_dir():
        raise DatasetNotFoundError(f"Invalid dataset root: {dataset_root}")

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    identity_map = {}

    for identity_dir in sorted(dataset_root.iterdir()):
        if identity_dir.is_dir():
            images = [p for p in sorted(identity_dir.iterdir()) if p.suffix.lower() in valid_extensions]
            
            if getattr(config, "MAX_IMAGES_PER_IDENTITY", None) is not None:
                images = images[:config.MAX_IMAGES_PER_IDENTITY]
                
            if len(images) >= 2:
                identity_map[identity_dir.name] = images

    if not identity_map:
        raise DatasetNotFoundError("No valid identities found.")

    logger.info(f"Discovered {len(identity_map)} identities.")
    return identity_map

def split_gallery_probe(identity_map: dict[str, list[Path]], gallery_ratio: float = config.GALLERY_RATIO, seed: int = config.RANDOM_SEED):
    logger.info(f"Splitting data (Gallery: {gallery_ratio*100:.0f}%, Probe: {(1-gallery_ratio)*100:.0f}%)")
    rng = random.Random(seed)
    gallery_map, probe_map = {}, {}

    for identity, images in identity_map.items():
        shuffled = images.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * gallery_ratio))
        gallery_map[identity] = shuffled[:split_idx]
        probe_map[identity] = shuffled[split_idx:]

    return gallery_map, probe_map

def run_ingestion(embedder: BaseEmbedder | None = None, store: BaseVectorStore | None = None, dataset_root: Path = config.VGGFACE2_ROOT) -> None:
    logger.info("--- Starting Ingestion Pipeline ---")
    if embedder is None:
        from ..implementations.arcface_embedder import ArcFaceEmbedder
        embedder = ArcFaceEmbedder()
    if store is None:
        from ..implementations.faiss_vector_store import FaissVectorStore
        store = FaissVectorStore()

    identity_map = discover_dataset(dataset_root)
    gallery_map, probe_map = split_gallery_probe(identity_map)

    logger.info("Extracting embeddings for gallery images (with checkpointing)...")
    
    checkpoint_file = config.FAISS_INDEX_DIR / "processed_identities.json"
    processed_identities = set()
    
    # Load checkpoint if it exists
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                processed_list = json.load(f)
                processed_identities = set(processed_list)
            logger.info(f"Loaded checkpoint: {len(processed_identities)} identities already processed.")
            # Load the existing store to append to it
            try:
                store.load()
                logger.info(f"Loaded existing FAISS index with {store.total_vectors} vectors.")
            except Exception as e:
                logger.warning(f"Could not load FAISS index ({e}). Starting fresh.")
                processed_identities = set()
        except Exception as e:
            logger.warning(f"Could not read checkpoint file ({e}). Starting fresh.")

    # Sort items for deterministic processing
    for i, (identity, images) in enumerate(tqdm(gallery_map.items(), desc="Processing identities")):
        if identity in processed_identities:
            continue
            
        identity_embeddings, identity_metadata = [], []
        
        for img_path in images:
            try:
                emb = embedder.embed(str(img_path))
                if emb is not None:
                    identity_embeddings.append(emb)
                    identity_metadata.append({"identity": identity, "image_path": str(img_path)})
            except (ImageReadError, Exception):
                continue

        # Add this identity to the store immediately if valid faces were found
        if identity_embeddings:
            embeddings_matrix = np.stack(identity_embeddings, axis=0).astype(np.float32)
            store.add(embeddings_matrix, identity_metadata)
            
        processed_identities.add(identity)
        
        # Save checkpoint every 10 identities or at the very end
        if (i + 1) % 10 == 0 or (i + 1) == len(gallery_map):
            store.save()
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_identities), f)

    if store.total_vectors == 0:
        logger.error("No embeddings extracted. Aborting ingestion.")
        return

    probe_save_path = config.FAISS_INDEX_DIR / "probe_set.json"
    probe_serialisable = {ident: [str(p) for p in paths] for ident, paths in probe_map.items()}
    
    logger.info(f"Saving probe set to {probe_save_path}...")
    with open(probe_save_path, "w", encoding="utf-8") as f:
        json.dump(probe_serialisable, f, indent=2)

    logger.info(f"--- Ingestion complete. {store.total_vectors} vectors stored. ---")