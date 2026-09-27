import json
import logging
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from ..core import config
from ..interfaces.base_vector_store import BaseVectorStore
from ..implementations.multimodal_embedder import MultimodalEmbedder
from ..core.exceptions import DatasetNotFoundError, ImageReadError

logger = logging.getLogger(__name__)


def load_celeba_captions(caption_dir: Path) -> dict[str, str]:
    """
    Load CelebA-Dialog text captions from per-image .txt files.

    Expected structure:
        caption_dir/
        ├── 000001.txt   →  "This woman has wavy brown hair..."
        ├── 000002.txt
        └── ...

    Returns:
        dict mapping image stem (e.g. "000001") to its caption text.
    """
    if not caption_dir.exists():
        raise DatasetNotFoundError(f"Caption directory not found: {caption_dir}")

    captions = {}
    for txt_file in sorted(caption_dir.glob("*.txt")):
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            # Use just the first line/caption if multiple are present
            captions[txt_file.stem] = text.split("\n")[0].strip()

    logger.info(f"Loaded {len(captions)} captions from {caption_dir}")
    return captions


def load_celeba_identities(identity_file: Path) -> dict[str, str]:
    """
    Load CelebA identity labels from identity_CelebA.txt.

    Format per line: "000001.jpg 2880"

    Returns:
        dict mapping image stem (e.g. "000001") to identity string (e.g. "2880").
    """
    if not identity_file.exists():
        raise DatasetNotFoundError(f"Identity file not found: {identity_file}")

    identities = {}
    for line in identity_file.read_text(encoding="utf-8").strip().split("\n"):
        parts = line.strip().split()
        if len(parts) == 2:
            stem = Path(parts[0]).stem
            identities[stem] = parts[1]

    logger.info(f"Loaded {len(identities)} identity labels from {identity_file}")
    return identities


def discover_celeba_dataset(
    image_dir: Path,
    caption_dir: Path,
    identity_file: Path,
    max_per_identity: int | None = None,
) -> dict[str, list[dict]]:
    """
    Build a mapping of identity → list of {image_path, caption} dicts.

    Only includes images that have BOTH a caption and an identity label.
    """
    captions = load_celeba_captions(caption_dir)
    identities = load_celeba_identities(identity_file)

    valid_extensions = {".jpg", ".jpeg", ".png"}
    identity_map: dict[str, list[dict]] = {}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() not in valid_extensions:
            continue

        stem = img_path.stem
        if stem not in captions or stem not in identities:
            continue

        identity = identities[stem]
        entry = {"image_path": img_path, "caption": captions[stem]}

        if identity not in identity_map:
            identity_map[identity] = []
        identity_map[identity].append(entry)

    # Filter: need at least 2 images per identity for gallery/probe split
    identity_map = {k: v for k, v in identity_map.items() if len(v) >= 2}

    # Apply total identities limit (For CPU users)
    max_identities = getattr(config, "MAX_IDENTITIES", None)
    if max_identities is not None:
        identity_map = dict(list(identity_map.items())[:max_identities])

    # Apply per-identity limit
    if max_per_identity is not None:
        identity_map = {k: v[:max_per_identity] for k, v in identity_map.items()}

    if not identity_map:
        raise DatasetNotFoundError("No valid identities found with both captions and images.")

    total_images = sum(len(v) for v in identity_map.values())
    logger.info(f"Discovered {len(identity_map)} identities with {total_images} image-caption pairs.")
    return identity_map


def split_gallery_probe(
    identity_map: dict[str, list[dict]],
    gallery_ratio: float = config.GALLERY_RATIO,
    seed: int = config.RANDOM_SEED,
):
    rng = random.Random(seed)
    gallery_map, probe_map = {}, {}

    for identity, entries in identity_map.items():
        shuffled = entries.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, int(len(shuffled) * gallery_ratio))
        gallery_map[identity] = shuffled[:split_idx]
        probe_map[identity] = shuffled[split_idx:]

    return gallery_map, probe_map


def run_multimodal_ingestion(
    mm_embedder: MultimodalEmbedder,
    store: BaseVectorStore,
    store_dir: Path,
    image_dir: Path = None,
    caption_dir: Path = None,
    identity_file: Path = None,
) -> None:
    logger.info(f"--- Starting Multimodal Ingestion Pipeline (Saving to {store_dir}) ---")

    # Use config defaults if not provided
    image_dir = image_dir or config.CELEBA_IMAGE_DIR
    caption_dir = caption_dir or config.CELEBA_CAPTION_DIR
    identity_file = identity_file or config.CELEBA_IDENTITY_FILE

    identity_map = discover_celeba_dataset(
        image_dir, caption_dir, identity_file,
        max_per_identity=getattr(config, "MAX_IMAGES_PER_IDENTITY", None),
    )
    gallery_map, probe_map = split_gallery_probe(identity_map)

    logger.info("Extracting multimodal embeddings for gallery (with checkpointing)...")

    checkpoint_file = store_dir / "processed_identities.json"
    processed_identities = set()

    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r") as f:
                processed_identities = set(json.load(f))
            logger.info(f"Loaded checkpoint: {len(processed_identities)} identities already processed.")
            try:
                store.load(store_dir)
                logger.info(f"Loaded existing index with {store.total_vectors} vectors.")
            except Exception as e:
                logger.warning(f"Could not load index ({e}). Starting fresh.")
                processed_identities = set()
        except Exception as e:
            logger.warning(f"Could not read checkpoint ({e}). Starting fresh.")

    for i, (identity, entries) in enumerate(tqdm(gallery_map.items(), desc="Processing identities")):
        if identity in processed_identities:
            continue

        identity_embeddings, identity_metadata = [], []

        for entry in entries:
            try:
                emb = mm_embedder.embed(str(entry["image_path"]), entry["caption"])
                if emb is not None:
                    identity_embeddings.append(emb)
                    identity_metadata.append({
                        "identity": identity,
                        "image_path": str(entry["image_path"]),
                        "caption": entry["caption"],
                    })
            except (ImageReadError, Exception) as e:
                logger.debug(f"Skipping {entry['image_path']}: {e}")
                continue

        if identity_embeddings:
            embeddings_matrix = np.stack(identity_embeddings, axis=0).astype(np.float32)
            store.add(embeddings_matrix, identity_metadata)

        processed_identities.add(identity)

        if (i + 1) % 10 == 0 or (i + 1) == len(gallery_map):
            store.save(store_dir)
            with open(checkpoint_file, "w") as f:
                json.dump(list(processed_identities), f)

    if store.total_vectors == 0:
        logger.error("No embeddings extracted. Aborting.")
        return

    # Save probe set (with captions included for multimodal evaluation)
    probe_save_path = store_dir / "probe_set.json"
    probe_serialisable = {
        identity: [
            {"image_path": str(e["image_path"]), "caption": e["caption"]}
            for e in entries
        ]
        for identity, entries in probe_map.items()
    }

    with open(probe_save_path, "w", encoding="utf-8") as f:
        json.dump(probe_serialisable, f, indent=2)

    logger.info(f"--- Multimodal ingestion complete. {store.total_vectors} vectors stored. ---")
