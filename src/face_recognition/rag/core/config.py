from pathlib import Path

PROJECT_ROOT = Path(r"C:\codingace_recognition")
DATASET_ROOT = PROJECT_ROOT / "src" / "dataset"
VGGFACE2_ROOT = DATASET_ROOT / "vggface2_test"
FAISS_INDEX_DIR = PROJECT_ROOT / "src" / "face_recognition" / "rag" / "faiss_store"

ARCFACE_MODEL_NAME = "buffalo_l"
ARCFACE_PROVIDERS = ["CPUExecutionProvider"]
DET_SIZE = (640, 640)
EMBEDDING_DIM = 512

GALLERY_RATIO = 0.80
SIMILARITY_THRESHOLD = 0.28
TOP_K = 5
RANDOM_SEED = 42

# Testing limits: Set to an integer (e.g., 20) for fast testing, or None for full dataset.
MAX_IMAGES_PER_IDENTITY = 20
