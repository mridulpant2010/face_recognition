import json
import logging
from pathlib import Path

import numpy as np

from constants import (
    ALLOWED_IMAGE_EXTENSIONS,
    CUSTOM_FACES_ROOT,
    EMBEDDING_DIM,
    UNIMODAL_INDEX_DIR,
)
from rag.implementations.arcface_embedder import ArcFaceEmbedder
from rag.implementations.faiss_vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

embedder = ArcFaceEmbedder()
vector_store = FaissVectorStore(dim=EMBEDDING_DIM)
vector_store.load()


def search_for_unprocessed_images(base_path: Path = CUSTOM_FACES_ROOT):

    base_train_path = base_path / "train"
    metadata_filepath = UNIMODAL_INDEX_DIR / "metadata.json"
    all_existing_images = []

    for file in base_train_path.rglob("*"):
        if file.is_file() and file.suffix.lower() in ALLOWED_IMAGE_EXTENSIONS:
            all_existing_images.append(str(file))

    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            processed_images_metadata = json.load(f)
    except FileNotFoundError as e:
        print(f"No metadata file found. Error: {e}")
        processed_images_metadata = {}

    processed_images = processed_images_metadata.keys()

    unprocessed_images = set(all_existing_images).difference(set(processed_images))

    return list(unprocessed_images)


def process_new_images(image_filepaths: list[Path]):

    metadata_filepath = UNIMODAL_INDEX_DIR / "metadata.json"
    unique_identities = 0

    try:
        with open(metadata_filepath, "r", encoding="utf-8") as f:
            processed_images_metadata = json.load(f)
            unique_identities: int = processed_images_metadata["unique_identities"]
    except FileNotFoundError as e:
        print(f"No metadata file found. Error: {e}")

    for file_path in image_filepaths:
        logger.info("-" * 20)
        logger.info(f"Processing image file: {file_path}")

        image_faces_metadata: list[dict[str, str | int | Path]] = []

        all_face_embeddings = embedder.embed_all_faces(file_path)
        all_face_embeddings_norm = []

        for i, face_embedding in enumerate(all_face_embeddings):
            logger.info(f"Processing face index {i}")

            previously_identified_face: bool = False
            face_metadata = {}

            face_metadata["image_file"] = str(file_path)

            face_embedding = face_embedding / np.linalg.norm(face_embedding)
            face_similarity_results = vector_store.search(face_embedding, top_k=1)
            logger.info(face_similarity_results)

            if (
                len(face_similarity_results) != 0
                and face_similarity_results[0]["similarity"] >= 0.45
            ):
                previously_identified_face = True

            if previously_identified_face:
                logger.info(
                    f"Previously identified face id: {face_similarity_results[0]['person_label']} encountered"
                )
                face_metadata["person_label"] = face_similarity_results[0][
                    "person_label"
                ]
            else:
                logger.info("New face identified")
                face_metadata["person_label"] = unique_identities
                unique_identities = unique_identities + 1

            image_faces_metadata.append(face_metadata)

            face_embedding = face_embedding / np.linalg.norm(x=face_embedding)
            all_face_embeddings_norm.append(face_embedding)

        all_face_embeddings_norm = np.stack(all_face_embeddings_norm, axis=0).astype(
            np.float32
        )

        vector_store.add(all_face_embeddings_norm, image_faces_metadata)

    vector_store.save()


if __name__ == "__main__":
    unprocessed_images = search_for_unprocessed_images()
    logger.info(f"Unprocessed images: {len(unprocessed_images)}")

    process_new_images(unprocessed_images)
