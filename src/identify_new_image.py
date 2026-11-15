import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from constants import EMBEDDING_DIM
from rag.implementations.faiss_vector_store import FaissVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def identify_and_draw(test_image_path: str, threshold: float = 0.60):
    logger.info("Loading FAISS Vector Store...")
    vector_store = FaissVectorStore(dim=EMBEDDING_DIM)
    vector_store.load()

    logger.info("Loading InsightFace Model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640, 640))

    img = cv2.imread(test_image_path)
    if img is None:
        logger.info(f"Error: Could not read image at {test_image_path}")
        return

    faces = app.get(img)
    logger.info(f"Detected {len(faces)} faces.")

    for face in faces:
        bbox = face.bbox.astype(int)
        kps = face.kps.astype(int)

        face_embedding = face.embedding / np.linalg.norm(face.embedding)
        face_similarity_results = vector_store.search(face_embedding, top_k=1)

        identity_label = "Unknown"
        color = (0, 0, 255)

        if face_similarity_results:
            best_match = face_similarity_results[0]
            if best_match["similarity"] >= threshold:
                person_id = best_match["person_label"]
                similarity_score = best_match["similarity"]
                identity_label = f"ID: {person_id} ({similarity_score:.2f})"
                color = (0, 255, 0)

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        for kp in kps:
            cv2.circle(img, tuple(kp), radius=2, color=(0, 255, 255), thickness=-1)

        text_position = (bbox[0], max(0, bbox[1] - 10))
        cv2.putText(
            img,
            identity_label,
            text_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

    cv2.namedWindow("Face Identification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Identification", 1280, 720)

    cv2.imshow("Face Identification", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    TEST_IMAGE = "dataset/shubham/train/IMG-20250418-WA0022.jpg"
    identify_and_draw(TEST_IMAGE, threshold=0.45)
