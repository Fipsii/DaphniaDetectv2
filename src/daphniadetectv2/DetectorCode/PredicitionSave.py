import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def save_yolo_annotation(label_path, results, image_shape):
    """Save YOLO segmentation annotations in the correct format."""
    h, w = image_shape[:2]

    with open(label_path, "w") as f:
        for r in results:
            for mask, box in zip(r.masks.xy, r.boxes):
                class_id = int(box.cls.item())  # Class ID
                normalized_points = [(x / w, y / h) for x, y in mask]  # Normalize coordinates
                
                # Format as YOLO annotation
                points_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized_points)
                f.write(f"{class_id} {points_str}\n")

