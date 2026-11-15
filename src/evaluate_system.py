import json
from pathlib import Path

import bcubed
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure

from constants import EMBEDDING_DIM
from rag.implementations.faiss_vector_store import FaissVectorStore


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xB < xA or yB < yA:
        return 0.0

    interArea = (xB - xA) * (yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


def evaluate_system(json_filepath: str, images_dir: Path, threshold: float = 0.60):
    print("Loading FAISS Vector Store...")
    vector_store = FaissVectorStore(dim=EMBEDDING_DIM)
    vector_store.load()

    print("Loading InsightFace Model...")
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(640, 640))

    with open(json_filepath, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    ldict_gt = {}
    ldict_pred = {}
    aligned_gt = []
    aligned_pred = []

    total_images = 0
    matched_faces = 0

    for task in tasks:
        image_filename = task.get("file_upload", "")

        if "-" in image_filename[:15]:
            local_filename = image_filename.split("-", 1)[-1]
        else:
            local_filename = image_filename

        img_path = images_dir / local_filename
        if not img_path.exists():
            print(f"Skipping {local_filename} - Image not found locally.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        total_images += 1

        gt_faces = []
        annotations = task.get("annotations", [])
        if not annotations:
            continue

        results = annotations[0].get("result", [])
        grouped_results = {}

        for res in results:
            f_id = res["id"]
            if f_id not in grouped_results:
                grouped_results[f_id] = {}

            if res["type"] == "rectanglelabels":
                val = res["value"]
                w_orig, h_orig = res["original_width"], res["original_height"]

                x1 = (val["x"] / 100.0) * w_orig
                y1 = (val["y"] / 100.0) * h_orig
                w = (val["width"] / 100.0) * w_orig
                h = (val["height"] / 100.0) * h_orig
                grouped_results[f_id]["bbox"] = [x1, y1, x1 + w, y1 + h]

            elif res["type"] == "textarea":
                grouped_results[f_id]["label"] = res["value"]["text"][0]

        for f_id, data in grouped_results.items():
            if "bbox" in data and "label" in data:
                gt_faces.append(data)

        predicted_faces = app.get(img)

        for gt_idx, gt in enumerate(gt_faces):
            best_iou = 0
            best_pred = None

            for pred in predicted_faces:
                iou = calculate_iou(gt["bbox"], pred.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_pred = pred

            # If IoU > 0.3, we consider the face successfully detected and matched
            if best_iou > 0.3 and best_pred is not None:
                matched_faces += 1

                face_embedding = best_pred.embedding / np.linalg.norm(
                    best_pred.embedding
                )
                similarity_results = vector_store.search(face_embedding, top_k=1)

                pred_label = "Unknown"
                if (
                    similarity_results
                    and similarity_results[0]["similarity"] >= threshold
                ):
                    pred_label = str(similarity_results[0]["person_label"])

                gt_label = str(gt["label"])

                unique_item_id = f"{local_filename}_face_{gt_idx}"
                ldict_gt[unique_item_id] = set([gt_label])
                ldict_pred[unique_item_id] = set([pred_label])

                aligned_gt.append(gt_label)
                aligned_pred.append(pred_label)

    print("\n" + "=" * 40)
    print(
        f"Evaluation Results (Images: {total_images}, Matched Faces: {matched_faces})"
    )
    print("=" * 40)

    if matched_faces == 0:
        print("No faces were matched. Check image paths and detection thresholds.")
        return

    precision = bcubed.precision(ldict_pred, ldict_gt)
    recall = bcubed.recall(ldict_pred, ldict_gt)
    f1 = bcubed.fscore(precision, recall)

    print(f"B-Cubed Precision : {precision:.3f}")
    print(f"B-Cubed Recall    : {recall:.3f}")
    print(f"B-Cubed F1 Score  : {f1:.3f}")

    ari_score = adjusted_rand_score(aligned_gt, aligned_pred)
    homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(
        aligned_gt, aligned_pred
    )

    print("-" * 40)
    print(f"Adjusted Rand Index (ARI) : {ari_score:.3f}")
    print(f"Homogeneity               : {homogeneity:.3f}")
    print(f"Completeness              : {completeness:.3f}")
    print(f"V-Measure                 : {v_measure:.3f}")


if __name__ == "__main__":
    JSON_FILE = "custom_data_label_ground_truth.json"
    IMAGES_DIR = Path("dataset/shubham/train")

    evaluate_system(json_filepath=JSON_FILE, images_dir=IMAGES_DIR, threshold=0.60)
