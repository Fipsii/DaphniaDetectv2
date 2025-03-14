import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO(OutputFolder)  # Change to your model file

# Paths
input_images_folder =  # Folder with images to predict
output_images_folder = 
output_labels_folder = 

# Create output directories if they don’t exist
os.makedirs(output_images_folder, exist_ok=True)
os.makedirs(output_labels_folder, exist_ok=True)

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

# Run inference on images
image_files = [f for f in os.listdir(input_images_folder) if f.endswith((".jpg", ".png"))]

for image_file in image_files:
    image_path = os.path.join(input_images_folder, image_file)
    image = cv2.imread(image_path)
    
    # Run YOLOv11 inference
    results = model(image)

    # Save prediction image
    output_image_path = os.path.join(output_images_folder, image_file)
    cv2.imwrite(output_image_path, image)

    # Save annotations in YOLO format
    output_label_path = os.path.join(output_labels_folder, os.path.splitext(image_file)[0] + ".txt")
    save_yolo_annotation(output_label_path, results, image.shape)
