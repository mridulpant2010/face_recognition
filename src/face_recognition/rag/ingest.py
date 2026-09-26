"""
Ingestion Pipeline
==================
Phase 1 of the RAG pipeline.

This module:
    1. Scans the VGGFace2 test directory to discover all identities and images.
    2. Splits each identity's images into Gallery (80%) and Probe (20%).
    3. Embeds every Gallery image using ArcFace.
    4. Stores the embeddings + metadata in the FAISS vector store.
    5. Saves the Probe image paths to a JSON file for later evaluation.

VGGFace2 test folder structure:
    vggface2_test/
        n000001/         ← Identity 1
            0001_01.jpg
            0002_01.jpg
            ...
        n000002/         ← Identity 2
            0001_01.jpg
            ...

After running this module, you will have:
    - A populated FAISS index on disk (faiss_store/index.faiss)
    - A metadata JSON mapping each vector to its identity + source image
    - A probe set JSON listing all held-out query images for evaluation
"""

import json
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

from . import config
from .embedder import ArcFaceEmbedder
from .vector_store import FaissVectorStore


def discover_dataset(dataset_root: Path) -> dict[str, list[Path]]:
    """
    Scan the VGGFace2 directory and return a mapping of
    identity_name → list of image file paths.

    Args:
        dataset_root: Path to the vggface2_test/ folder.

    Returns:
        dict like {"n000001": [Path("0001_01.jpg"), ...], ...}
    """
    identity_map = {}
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for identity_dir in sorted(dataset_root.iterdir()):
        if not identity_dir.is_dir():
            continue

        images = [
            img_path
            for img_path in sorted(identity_dir.iterdir())
            if img_path.suffix.lower() in valid_extensions
        ]

        if len(images) >= 2:
            # Need at least 2 images: 1 for gallery, 1 for probe
            identity_map[identity_dir.name] = images

    return identity_map


def split_gallery_probe(
    identity_map: dict[str, list[Path]],
    gallery_ratio: float = config.GALLERY_RATIO,
    seed: int = config.RANDOM_SEED,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """
    Split each identity's images into Gallery and Probe sets.

    Args:
        identity_map:  Output of discover_dataset().
        gallery_ratio: Fraction of images per identity that go into the gallery.
        seed:          Random seed for reproducibility.

    Returns:
        (gallery_map, probe_map) — same structure as identity_map.
    """
    rng = random.Random(seed)
    gallery_map = {}
    probe_map = {}

    for identity, images in identity_map.items():
        shuffled = images.copy()
        rng.shuffle(shuffled)

        split_idx = max(1, int(len(shuffled) * gallery_ratio))
        gallery_map[identity] = shuffled[:split_idx]
        probe_map[identity] = shuffled[split_idx:]

    return gallery_map, probe_map


def run_ingestion(dataset_root: Path = config.VGGFACE2_ROOT) -> None:
    """
    Full ingestion pipeline:
        1. Discover identities in VGGFace2.
        2. Split into gallery/probe.
        3. Embed all gallery images with ArcFace.
        4. Store in FAISS.
        5. Save probe set for evaluation.
    """
    print("=" * 60)
    print("FACE RAG — INGESTION PIPELINE")
    print("=" * 60)

    # --- Step 1: Discover the dataset ---
    print(f"\n[Step 1] Scanning dataset at: {dataset_root}")
    identity_map = discover_dataset(dataset_root)
    total_images = sum(len(imgs) for imgs in identity_map.values())
    print(f"  Found {len(identity_map)} identities, {total_images} total images.")

    if not identity_map:
        print("  ERROR: No identities found. Check your VGGFACE2_ROOT path in config.py.")
        return

    # --- Step 2: Split into Gallery and Probe ---
    print(f"\n[Step 2] Splitting into Gallery ({config.GALLERY_RATIO:.0%}) / Probe ({1 - config.GALLERY_RATIO:.0%})")
    gallery_map, probe_map = split_gallery_probe(identity_map)

    gallery_count = sum(len(imgs) for imgs in gallery_map.values())
    probe_count = sum(len(imgs) for imgs in probe_map.values())
    print(f"  Gallery: {gallery_count} images across {len(gallery_map)} identities")
    print(f"  Probe  : {probe_count} images across {len(probe_map)} identities")

    # --- Step 3: Embed all Gallery images ---
    print(f"\n[Step 3] Embedding gallery images with ArcFace ({config.ARCFACE_MODEL_NAME})...")
    embedder = ArcFaceEmbedder()
    store = FaissVectorStore()

    all_embeddings = []
    all_metadata = []
    failed_count = 0

    for identity, images in tqdm(gallery_map.items(), desc="Identities"):
        for img_path in images:
            emb = embedder.embed(str(img_path))
            if emb is not None:
                all_embeddings.append(emb)
                all_metadata.append({
                    "identity": identity,
                    "image_path": str(img_path),
                })
            else:
                failed_count += 1

    print(f"  Successfully embedded: {len(all_embeddings)}")
    print(f"  Failed (no face detected): {failed_count}")

    # --- Step 4: Store in FAISS ---
    print(f"\n[Step 4] Adding {len(all_embeddings)} vectors to FAISS index...")
    embeddings_matrix = np.stack(all_embeddings, axis=0).astype(np.float32)
    store.add(embeddings_matrix, all_metadata)
    store.save()

    # --- Step 5: Save the Probe set for evaluation ---
    print(f"\n[Step 5] Saving probe set...")
    probe_save_path = config.FAISS_INDEX_DIR / "probe_set.json"
    config.FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Convert Path objects to strings for JSON serialisation
    probe_serialisable = {
        identity: [str(p) for p in paths]
        for identity, paths in probe_map.items()
    }
    with open(probe_save_path, "w") as f:
        json.dump(probe_serialisable, f, indent=2)
    print(f"  Probe set saved → {probe_save_path}")

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  FAISS index : {config.FAISS_INDEX_DIR / 'index.faiss'}")
    print(f"  Metadata    : {config.FAISS_INDEX_DIR / 'metadata.json'}")
    print(f"  Probe set   : {probe_save_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_ingestion()
