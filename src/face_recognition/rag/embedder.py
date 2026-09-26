"""
ArcFace Embedder
================
Wraps the InsightFace ArcFace pipeline into a simple, reusable class.
Handles: image loading → face detection → alignment → 512-D embedding extraction.

This is the "Feature Extraction" step of the RAG pipeline.
Every image that enters the system (whether for ingestion or querying)
passes through this module.
"""

import cv2
import numpy as np
from pathlib import Path
from insightface.app import FaceAnalysis

from . import config


class ArcFaceEmbedder:
    """
    Singleton-style wrapper around InsightFace's FaceAnalysis.

    Usage:
        embedder = ArcFaceEmbedder()
        embedding = embedder.embed("path/to/face.jpg")
        # Returns: np.ndarray of shape (512,), L2-normalised. Or None if no face found.
    """

    def __init__(self):
        print("[Embedder] Loading ArcFace model...")
        self.app = FaceAnalysis(
            name=config.ARCFACE_MODEL_NAME,
            providers=config.ARCFACE_PROVIDERS,
        )
        self.app.prepare(ctx_id=0, det_size=config.DET_SIZE)
        print("[Embedder] ArcFace model ready.")

    def embed(self, image_path: str) -> np.ndarray | None:
        """
        Extract the 512-D ArcFace embedding from the largest face in the image.

        Steps:
            1. Load image with OpenCV (BGR format).
            2. Run RetinaFace detection to find all faces + 5-point landmarks.
            3. InsightFace internally performs affine alignment to 112×112.
            4. ArcFace R100 backbone produces a 512-D vector.
            5. The vector is L2-normalised so cosine similarity = dot product.

        Returns:
            np.ndarray of shape (512,) if a face is found, else None.
        """
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return None

        faces = self.app.get(img_bgr)
        if not faces:
            return None

        # Pick the face with the highest detection confidence
        best_face = max(faces, key=lambda f: f.det_score)
        return best_face.normed_embedding  # shape: (512,), L2-normalised

    def embed_batch(self, image_paths: list[str]) -> list[tuple[str, np.ndarray]]:
        """
        Embed a list of images. Returns a list of (image_path, embedding) tuples,
        skipping any images where no face was detected.
        """
        results = []
        for path in image_paths:
            emb = self.embed(path)
            if emb is not None:
                results.append((str(path), emb))
        return results
