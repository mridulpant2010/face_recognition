"""
Configuration for the Face Recognition RAG Pipeline
====================================================
All paths, thresholds, and constants live here so every
other module imports from a single source of truth.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(r"C:\coding\face_recognition")
DATASET_ROOT = PROJECT_ROOT / "src" / "dataset"

# VGGFace2 test split folder structure expected:
#   VGGFACE2_ROOT/
#     n000001/
#       0001_01.jpg
#       0002_01.jpg
#       ...
#     n000002/
#       ...
# Each subfolder name (e.g. n000001) is one identity.
VGGFACE2_ROOT = DATASET_ROOT / "vggface2_test"

# Where the FAISS index and metadata are persisted to disk
FAISS_INDEX_DIR = PROJECT_ROOT / "src" / "face_recognition" / "rag" / "faiss_store"

# ---------------------------------------------------------------------------
# ArcFace Model Settings
# ---------------------------------------------------------------------------
ARCFACE_MODEL_NAME = "buffalo_l"              # RetinaFace detector + ArcFace R100
ARCFACE_PROVIDERS  = ["CPUExecutionProvider"]  # Swap to ["CUDAExecutionProvider"] for GPU
DET_SIZE           = (640, 640)               # Resolution for face detection
EMBEDDING_DIM      = 512                      # ArcFace outputs 512-D vectors

# ---------------------------------------------------------------------------
# RAG Pipeline Settings
# ---------------------------------------------------------------------------
# Gallery/Probe split ratio: 80% of each identity's images go into the
# gallery (the vector database), 20% are held out as probe queries.
GALLERY_RATIO = 0.80

# Cosine similarity threshold for a positive match.
# Vectors below this score are classified as "Unknown".
SIMILARITY_THRESHOLD = 0.28

# Number of nearest neighbours to retrieve per query
TOP_K = 5

# Random seed for reproducible gallery/probe splits
RANDOM_SEED = 42
