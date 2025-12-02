import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO

DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Step 1: Masking
def mask_pcb():
    img_path = r"C:\Users\ranja\Downloads\Project 3\Project 3 Data\motherboard_image.JPEG"  
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    blur = cv2.GaussianBlur(img_rot, (47, 47), 4)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY, 55, 7
    )
    edges = cv2.Canny(th, 50, 350)
    edges_d = cv2.dilate(edges, None, iterations=12)

    contours, _ = cv2.findContours(edges_d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found")

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(img_rot)
    cv2.drawContours(mask, [largest], -1, (255, 255, 255), cv2.FILLED)

    extracted = cv2.bitwise_and(img_rot, mask)

    out_dir = Path("step1_outputs")
    out_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(out_dir / "edges.png"), edges)
    cv2.imwrite(str(out_dir / "mask.png"), mask)
    cv2.imwrite(str(out_dir / "extracted_pcb.png"), extracted)

    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(edges_rgb)
    plt.axis("off")
    plt.title("Edges")

    plt.subplot(1, 3, 2)
    plt.imshow(mask_rgb)
    plt.axis("off")
    plt.title("Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(extracted_rgb)
    plt.axis("off")
    plt.title("Extracted PCB")

    plt.tight_layout()
    plt.show()


# Step 2: Training
def train_yolo():
    data_yaml = r"C:\Users\ranja\Downloads\Project 3\Project 3 Data\data\data.yaml"
    data_yaml = str(Path(data_yaml).resolve())
    model = YOLO("yolo11n.pt")
    model.train(
        data=data_yaml,
        epochs=175,
        imgsz=1200,
        batch=3,
        workers=0,
        device=DEVICE,
        name="yolo11_pcb_project",
    )


def plot_curves():
    run_dir = Path("runs/detect") / "yolo11_pcb_project"
    figs = [
        ("P_curve.png", "Precision–Confidence"),
        ("PR_curve.png", "Precision–Recall"),
        ("confusion_matrix_normalized.png", "Confusion Matrix"),
    ]
    for fname, title in figs:
        path = (run_dir / fname).resolve()
        if not path.exists():
            continue
        img = mpimg.imread(str(path))
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis("off")
        plt.title(title)
        plt.show()


# Step 3: Evaluation
def eval_model():

    weights = Path(r"C:\Users\ranja\Downloads\Project 3\runs\detect\yolo11_pcb_project3\weights\best.pt")

    if not weights.exists():
        raise FileNotFoundError(weights)

    print("\nLoading model from:", weights)
    model = YOLO(str(weights))

    eval_dir = Path(r"C:\Users\ranja\Downloads\Project 3\Project 3 Data\data\evaluation")
    image_files = list(eval_dir.glob("*.jpg"))
    print("\nFound evaluation images:", image_files)

    if len(image_files) == 0:
        raise RuntimeError("No JPG images found in evaluation folder!")

    results = model.predict(
        source=[str(p) for p in image_files],
        conf=0.45,
        save=True,
        save_txt=False,
    )

    print("\nEvaluation complete! Check runs/detect/predict for output images.\n")

    for p in image_files:
        pred_path = Path("runs/detect/predict") / p.name
        if pred_path.exists():
            img_bgr = cv2.imread(str(pred_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 6))
            plt.imshow(img_rgb)
            plt.axis("off")
            plt.title(f"Predicted — {p.name}")
            plt.show()


if __name__ == "__main__":
    train_yolo()        
    plot_curves()      
    eval_model()         