# Face Recognition & Identity Verification RAG Pipeline

This repository contains a robust Retrieval-Augmented Generation (RAG) pipeline for Face Identity Verification. It supports both **Unimodal** (Image-only) and **Multimodal** (Image + Text) embedding architectures.

The codebase is built following strict **SOLID principles**, making it highly modular and easy for researchers and developers to extend.

## 📦 Dependencies & Technologies
This project relies on the following core libraries:
* **`insightface`** & **`onnxruntime`**: For extracting 512-D facial feature embeddings (ArcFace).
* **`sentence-transformers`**: For extracting 512-D natural language text embeddings (CLIP).
* **`faiss-cpu`**: For blazing-fast vector similarity search and retrieval.
* **`torch`** & **`torchvision`**: Core deep learning backend.

## ⚙️ Installation

1. Ensure you have Python 3.12+ installed.
2. Clone this repository and navigate to the root folder:
   ```bash
   git clone https://github.com/yourusername/face_recognition.git
   cd face_recognition
   ```
3. Install the required dependencies using the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *(Alternatively, you can install the package in editable mode: `pip install -e .`)*

## 💻 How to Use the Code (Python API)
If you want to utilize the modules in your own custom scripts, the codebase is designed to be plug-and-play.

### Example: Extracting a Multimodal Embedding
```python
from src.face_recognition.rag.implementations.arcface_embedder import ArcFaceEmbedder
from src.face_recognition.rag.implementations.text_encoder import SentenceTransformerEncoder
from src.face_recognition.rag.implementations.multimodal_embedder import MultimodalEmbedder

# 1. Initialize the individual towers
image_embedder = ArcFaceEmbedder()
text_encoder = SentenceTransformerEncoder(model_name="clip-ViT-B-32")

# 2. Initialize the Multimodal Fusion Embedder
mm_embedder = MultimodalEmbedder(
    image_embedder=image_embedder,
    text_encoder=text_encoder,
    fusion="concat" # Options: 'concat', 'add', 'weighted_add'
)

# 3. Extract a fused 1024-D identity vector!
vector = mm_embedder.embed(
    image_path="path/to/face.jpg", 
    text="A smiling woman with blonde hair."
)
print(vector.shape) # Output: (1024,)
```

---

## 📂 Dataset Setup (For running the built-in Pipeline)

If you want to run the provided evaluation pipelines, place your datasets in the correct folder structure.

### 1. Unimodal Dataset (VGGFace2)
Place the VGGFace2 test split here:
```text
src/dataset/vggface2_test/
├── n000001/
│   ├── 01.jpg
│   └── 02.jpg
```

### 2. Multimodal Dataset (CelebA-Dialog)
Place the CelebA-Dialog dataset components here:
```text
src/dataset/celeba_dialog/
├── img_align_celeba/          # The face images
├── celeba-caption/            # The .txt files with descriptions
└── identity_CelebA.txt        # The file linking images to IDs
```

> **Note for CPU Users:** By default, `config.py` is set to `MAX_IDENTITIES = 50` so that testing runs quickly (in ~2 minutes) on a standard laptop CPU. To process the entire dataset, change this to `None` in `src/face_recognition/rag/core/config.py`.

---

## 🚀 Running the Pipeline (CLI)

All commands should be run from the root directory of the project.

### Unimodal Mode (Image Only)
* **Run the full pipeline (Ingest Gallery & Evaluate Probe):**
  ```bash
  python -m src.face_recognition.rag.run_pipeline
  ```
* **Query a specific image against the database:**
  ```bash
  python -m src.face_recognition.rag.run_pipeline --query "path/to/face.jpg"
  ```

### Multimodal Mode (Image + Text)
* **Run the full multimodal pipeline (Default: Concat fusion):**
  ```bash
  python -m src.face_recognition.rag.run_pipeline --multimodal
  ```
* **Experiment with different Fusion Strategies:**
  ```bash
  python -m src.face_recognition.rag.run_pipeline --multimodal --fusion add
  python -m src.face_recognition.rag.run_pipeline --multimodal --fusion weighted_add
  ```
* **Query using an Image AND Text:**
  ```bash
  python -m src.face_recognition.rag.run_pipeline --multimodal --query "path/to/face.jpg" --query-text "A young woman with blonde hair smiling."
  ```

## 📊 Outputs
The FAISS vector databases and evaluation `.json` reports are automatically saved in the root folder under `unimodal_embedding/` and `multimodal_embeddings/<fusion_strategy>/`.
