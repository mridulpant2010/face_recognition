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

    logger.info("Extracting embeddings for gallery images...")
    all_embeddings, all_metadata = [], []
    
    for identity, images in tqdm(gallery_map.items(), desc="Processing identities"):
        for img_path in images:
            try:
                emb = embedder.embed(str(img_path))
                if emb is not None:
                    all_embeddings.append(emb)
                    all_metadata.append({"identity": identity, "image_path": str(img_path)})
            except (ImageReadError, Exception):
                continue

    if not all_embeddings:
        logger.error("No embeddings extracted. Aborting ingestion.")
        return

    embeddings_matrix = np.stack(all_embeddings, axis=0).astype(np.float32)
    store.add(embeddings_matrix, all_metadata)
    store.save()

    probe_save_path = config.FAISS_INDEX_DIR / "probe_set.json"
    probe_serialisable = {ident: [str(p) for p in paths] for ident, paths in probe_map.items()}
    
    logger.info(f"Saving probe set to {probe_save_path}...")
    with open(probe_save_path, "w", encoding="utf-8") as f:
        json.dump(probe_serialisable, f, indent=2)

    logger.info(f"--- Ingestion complete. {store.total_vectors} vectors stored. ---")