def Classify_Species_old(Folder_With_Images, Classifier_Location):
    import os
    import numpy as np
    from ultralytics import YOLO

    # Load the YOLO model once
    model = YOLO(Classifier_Location)

    # Get all image paths directly from the folder
    image_paths = [os.path.join(Folder_With_Images, filename) 
                   for filename in os.listdir(Folder_With_Images) 
                   if filename.endswith(('.jpg', '.png'))]  # Only process images

    if not image_paths:
        print("No images found in the specified folder.")
        return {}

    # Class mapping
    class_labels = model.names

    # Process predictions one at a time
    results_data = {}
    for image_path in image_paths:
        result = model(image_path, imgsz=1280, verbose=False)[0]  # Run inference on a single image

        probs = result.probs.cpu().numpy()  # Convert to NumPy array

        predicted_class = np.argmax(probs.data) if np.max(probs.data) >= 0.75 else np.nan
        species = class_labels.get(predicted_class, "unknown")  # Get species name

        filename = os.path.basename(image_path)  # Extract filename
        results_data[filename] = species  # Store result in dictionary

    return results_data  # Return dictionary instead of DataFrame

import albumentations as A
from ultralytics import YOLO
import torch
from pathlib import Path
import cv2

# Transform is optional for inference; YOLO automatically resizes/normalizes


import os
import numpy as np
from ultralytics import YOLO
import albumentations as A
import torch

def Classify_Species(Folder_With_Images: str, Classifier_Location: str):
    """
    Classifies all images in a folder using a trained YOLOv11 classification model.
    Returns a dictionary {filename: predicted_class_name}, same as Classify_Species_old.
    """

    # Optional transforms (YOLO already handles resizing/normalization)
    microscopy_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Load YOLO model once
    model = YOLO(Classifier_Location)
    class_labels = model.names

    # Collect image paths
    image_paths = [
        os.path.join(Folder_With_Images, f)
        for f in os.listdir(Folder_With_Images)
        if f.lower().endswith((".jpg", ".png"))
    ]

    if not image_paths:
        print("⚠️ No images found in the specified folder.")
        return {}

    results_data = {}

    # Process each image individually
    for image_path in image_paths:
        results = model.predict(source=image_path, imgsz=640, conf=0.25, verbose=False)
        probs = results[0].probs

        class_index = int(probs.top1)
        confidence = float(probs.top1conf)

        # Only accept confident predictions (like old function)
        if confidence >= 0.75:
            species = class_labels[class_index]
        else:
            species = np.nan

        filename = os.path.basename(image_path)
        results_data[filename] = species

    return results_data



