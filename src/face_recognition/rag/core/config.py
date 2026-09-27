from pathlib import Path

PROJECT_ROOT = Path(r"C:\coding\face_recognition\src\face_recognition\rag")
DATASET_ROOT = PROJECT_ROOT / "dataset"
VGGFACE2_ROOT = DATASET_ROOT / "vggface2_test"
FAISS_INDEX_DIR = PROJECT_ROOT / "faiss_store"

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
MAX_IDENTITIES = 50  # Cap the number of people to process (Perfect for CPU testing)

# --- Output Directories ---
UNIMODAL_INDEX_DIR = PROJECT_ROOT / "unimodal_embedding"
MULTIMODAL_INDEX_DIR = PROJECT_ROOT / "multimodal_embeddings"

# Fallback for defaults in BaseVectorStore
FAISS_INDEX_DIR = UNIMODAL_INDEX_DIR

# --- Multimodal (CelebA-Dialog) Configuration ---
CELEBA_ROOT = DATASET_ROOT / "celeba_dialog"
CELEBA_IMAGE_DIR = CELEBA_ROOT / "img_align_celeba"
CELEBA_CAPTION_DIR = CELEBA_ROOT / "celeba-caption"
CELEBA_IDENTITY_FILE = CELEBA_ROOT / "identity_CelebA.txt"

TEXT_ENCODER_MODEL = "clip-ViT-B-32"
FUSION_STRATEGY = "concat"  # Options: "concat", "add", "weighted_add"
IMAGE_WEIGHT = 0.6  # Only used when FUSION_STRATEGY = "weighted_add"
