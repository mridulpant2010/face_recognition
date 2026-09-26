"""
ArcFace Face Verification with Detection Phase Visualization
============================================================
Model  : ArcFace (buffalo_l) via insightface
Detector: RetinaFace (built into insightface pipeline)
Backbone: ResNet-100 trained with ArcFace loss on MS1Mv3
Embedding: 512-D L2-normalised vector

Detection Phase Visualisation:
  - Panel 1: Original image with bounding box + 5-point landmarks
  - Panel 2: Aligned & cropped face chip (112×112) passed to ArcFace
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend - saves PNG instead of opening a window
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import insightface
from insightface.app import FaceAnalysis

# ---------------------------------------------------------------------------
# 1. Initialise the ArcFace pipeline
#    det_size  : resolution fed to RetinaFace detector (larger = slower but finds small faces)
#    providers : use CUDAExecutionProvider if GPU is available, else CPU
# ---------------------------------------------------------------------------
app = FaceAnalysis(
    name="buffalo_l",                       # det=RetinaFace, rec=ArcFace R100
    providers=["CPUExecutionProvider"]      # swap to ["CUDAExecutionProvider"] for GPU
)
app.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=0 → first GPU (ignored on CPU)

# ---------------------------------------------------------------------------
# 2. Detection phase visualisation
#    Shows everything the model sees before it computes any embedding
# ---------------------------------------------------------------------------
LANDMARK_LABELS = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
LANDMARK_COLORS = ["#00E5FF", "#00E5FF", "#FF4081", "#76FF03", "#76FF03"]
FACE_CHIP_SIZE  = 112   # ArcFace is trained on 112×112 aligned face chips


def visualise_detection(img_bgr: np.ndarray, faces: list, image_path: str) -> None:
    """
    Render a three-panel figure for the detection phase:
      Panel 1 - Original image with bounding box(es) and landmark dots
      Panel 2 - The 112×112 aligned face chip(s) that ArcFace receives
      Panel 3 - Landmark connectivity diagram on the chip
    Saves to <image_path_stem>_detection.png next to the source image.
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    n_faces = len(faces)

    fig = plt.figure(figsize=(6 + n_faces * 4, 5 * n_faces), facecolor="#0D1117")
    fig.suptitle(
        f"ArcFace · Detection Phase  —  {Path(image_path).name}  ({n_faces} face(s) found)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )

    for face_idx, face in enumerate(faces):
        row_offset = face_idx * 3   # 3 panels per face row

        # -- Panel 1: Original image with bounding box & landmarks --
        ax1 = fig.add_subplot(n_faces, 3, row_offset + 1)
        ax1.imshow(img_rgb)
        ax1.set_title(f"Face {face_idx+1} · Detection\n(RetinaFace bounding box + 5 landmarks)",
                      color="#58A6FF", fontsize=9)
        ax1.axis("off")

        # Bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)
        rect = patches.FancyBboxPatch(
            (x1, y1), x2 - x1, y2 - y1,
            boxstyle="round,pad=2",
            linewidth=2, edgecolor="#00E5FF", facecolor="none"
        )
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 8, f"conf: {face.det_score:.2f}",
                 color="#00E5FF", fontsize=8, fontweight="bold",
                 bbox=dict(facecolor="#0D1117", alpha=0.6, pad=1, edgecolor="none"))

        # 5-point landmarks (left_eye, right_eye, nose, left_mouth, right_mouth)
        if face.kps is not None:
            for (lx, ly), label, color in zip(face.kps, LANDMARK_LABELS, LANDMARK_COLORS):
                ax1.plot(lx, ly, "o", color=color, markersize=5, markeredgewidth=0.5,
                         markeredgecolor="white")
                ax1.annotate(label, (lx, ly), textcoords="offset points",
                             xytext=(4, 4), color=color, fontsize=6)
            # Draw facial geometry lines: eye-eye, eye-nose, nose-mouth
            kps = face.kps
            for (a, b) in [(0, 1), (0, 2), (1, 2), (2, 3), (2, 4)]:
                ax1.plot([kps[a][0], kps[b][0]], [kps[a][1], kps[b][1]],
                         "-", color="#FFFFFF", linewidth=0.7, alpha=0.5)

        # -- Panel 2: The aligned 112×112 face chip --
        ax2 = fig.add_subplot(n_faces, 3, row_offset + 2)
        # insightface stores the aligned chip in face.img (BGR, 112×112)
        if hasattr(face, "img") and face.img is not None:
            chip_rgb = cv2.cvtColor(face.img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: manually crop the bbox region
            chip_bgr = img_bgr[max(0, y1):y2, max(0, x1):x2]
            chip_rgb = cv2.cvtColor(
                cv2.resize(chip_bgr, (FACE_CHIP_SIZE, FACE_CHIP_SIZE)), cv2.COLOR_BGR2RGB
            )
        ax2.imshow(chip_rgb)
        ax2.set_title(f"Face {face_idx+1} · Aligned Chip\n({FACE_CHIP_SIZE}×{FACE_CHIP_SIZE}px — input to ArcFace R100)",
                      color="#58A6FF", fontsize=9)
        ax2.axis("off")

        # -- Panel 3: Embedding magnitude bar (first 32 dims shown) --
        ax3 = fig.add_subplot(n_faces, 3, row_offset + 3)
        if face.embedding is not None:
            emb = face.normed_embedding          # L2-normalised 512-D ArcFace vector
            ax3.bar(range(32), emb[:32],
                    color=["#00E5FF" if v > 0 else "#FF4081" for v in emb[:32]],
                    width=0.8, edgecolor="none")
            ax3.axhline(0, color="white", linewidth=0.5, alpha=0.4)
            ax3.set_facecolor("#161B22")
            ax3.tick_params(colors="gray", labelsize=7)
            ax3.set_title(f"Face {face_idx+1} · ArcFace Embedding\n(first 32 of 512 dims, L2-normalised)",
                          color="#58A6FF", fontsize=9)
            ax3.set_xlabel("Dimension index", color="gray", fontsize=7)
            ax3.set_ylabel("Value", color="gray", fontsize=7)
            for spine in ax3.spines.values():
                spine.set_edgecolor("#30363D")

    plt.tight_layout(pad=1.5)
    out_path = Path(image_path).parent / (Path(image_path).stem + "_detection.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [viz] Detection figure saved → {out_path}")


# ---------------------------------------------------------------------------
# 3. Embedding extraction with detection visualisation
# ---------------------------------------------------------------------------
def get_arcface_embedding(image_path: str, visualise: bool = True):
    """
    Runs the full ArcFace pipeline on image_path:
      1. Loads image with OpenCV (BGR)
      2. Detects faces with RetinaFace
      3. Optionally renders detection visualisation
      4. Returns the 512-D L2-normalised ArcFace embedding of the largest face
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print(f"\n[{Path(image_path).name}]")
    print(f"  Image shape  : {img_bgr.shape}  (H×W×C, BGR)")

    # RetinaFace detection + ArcFace embedding in one call
    faces = app.get(img_bgr)

    if not faces:
        print("  No face detected.")
        return None

    print(f"  Faces detected: {len(faces)}")
    for i, f in enumerate(faces):
        print(f"  Face {i+1}: bbox={f.bbox.astype(int).tolist()}, "
              f"det_score={f.det_score:.3f}, "
              f"embedding_norm={np.linalg.norm(f.embedding):.4f}")

    if visualise:
        visualise_detection(img_bgr, faces, image_path)

    # Use the face with the highest detection confidence
    best_face = max(faces, key=lambda f: f.det_score)
    return best_face.normed_embedding   # 512-D, L2-normalised


# ---------------------------------------------------------------------------
# 4. Verification
# ---------------------------------------------------------------------------
IMG1 = r'C:\coding\face_recognition\src\dataset\bikeride.jpg'
IMG2 = r'C:\coding\face_recognition\src\dataset\friends.jpg'

# ArcFace standard threshold: cosine similarity ≥ 0.28 → same person
# (derived from IJB-C/LFW benchmarks with buffalo_l)
ARCFACE_THRESHOLD = 0.28

emb1 = get_arcface_embedding(IMG1, visualise=True)
emb2 = get_arcface_embedding(IMG2, visualise=True)

print("\n" + "=" * 55)
print("ArcFace Verification Result")
print("=" * 55)

if emb1 is not None and emb2 is not None:
    # Cosine similarity on already L2-normalised vectors = dot product
    similarity = float(np.dot(emb1, emb2))
    print(f"  Cosine Similarity : {similarity:.4f}  (threshold ≥ {ARCFACE_THRESHOLD})")
    if similarity >= ARCFACE_THRESHOLD:
        print("  Verdict           : ✅ SAME PERSON")
    else:
        print("  Verdict           : ❌ DIFFERENT PEOPLE")
else:
    print("  Could not compare — one or both images had no detectable face.")
