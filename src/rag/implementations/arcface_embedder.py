import logging
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from src import constants

from ..core.exceptions import EmbeddingModelLoadError, ImageReadError
from ..interfaces.base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)


class ArcFaceEmbedder(BaseEmbedder):
    def __init__(
        self,
        model_name: str = constants.ARCFACE_MODEL_NAME,
        providers: list[str] | None = None,
        det_size: tuple[int, int] = constants.DET_SIZE,
    ):
        providers = providers or constants.ARCFACE_PROVIDERS
        logger.info(f"Initializing ArcFace model '{model_name}'...")
        try:
            self._app = FaceAnalysis(name=model_name, providers=providers)
            self._app.prepare(ctx_id=0, det_size=det_size)
            logger.info("ArcFace model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise EmbeddingModelLoadError(
                f"Failed to load ArcFace model '{model_name}': {e}"
            )

    def embed(self, image_path: str) -> np.ndarray | None:
        logger.debug(f"Embedding image: {image_path}")
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"Could not read image: {image_path}")
            raise ImageReadError(f"Could not read image: {image_path}")

        try:
            faces = self._app.get(img_bgr)
        except Exception as e:
            logger.error(f"Inference error on {image_path}: {e}")
            return None

        if not faces:
            logger.debug(f"No face detected in {image_path}")
            return None

        best_face = max(faces, key=lambda f: f.det_score)
        logger.debug(f"Face detected with confidence {best_face.det_score:.2f}")
        return best_face.normed_embedding

    def embed_batch(self, image_paths: list[str]) -> list[tuple[str, np.ndarray]]:
        logger.info(f"Embedding batch of {len(image_paths)} images...")
        results = []
        for path in image_paths:
            try:
                emb = self.embed(path)
                if emb is not None:
                    results.append((str(path), emb))
            except (ImageReadError, Exception):
                continue
        logger.info(
            f"Successfully embedded {len(results)}/{len(image_paths)} images."
        )
        return results

    def embed_all_faces(self, image_path: str | Path):
        logger.debug(f"Embedding image: {image_path}")

        all_face_embeddings = []

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            logger.warning(f"Could not read image: {image_path}")
            raise ImageReadError(f"Could not read image: {image_path}")

        try:
            faces = self._app.get(img_bgr)
        except Exception as e:
            logger.error(f"Inference error on {image_path}: {e}")
            return

        if not faces:
            logger.debug(f"No face detected in {image_path}")
            return

        for face in faces:
            all_face_embeddings.append(np.array(face.embedding, dtype=np.float32))

        logger.info(f"Detected {len(all_face_embeddings)} faces in {image_path}")

        return all_face_embeddings
