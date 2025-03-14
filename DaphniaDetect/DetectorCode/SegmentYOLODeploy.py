import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO

def Segment(ImageDir, OutputDir, ModelPath, Vis=True):
    
    # Load the YOLOv11 segmentation model
    model = YOLO(ModelPath) 
    
    # Run inference on all images first
    results = model(ImageDir, imgsz=1280, save_txt = True,project=OutputDir, name="Segmentation", save = Vis, iou = 0.1, verbose=False)
    # Process results generator
    for result in results:
        masks = result.masks  # Masks object for segmentation masks output
        #print(masks)
       



import os
import cv2
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np

def Segment_Exp(ImageDir, OutputDir,ModelPath, Vis=True):
    """
    Run segmentation on images using YOLOv11 (or another variant).
    Saves the segmentation results to the OutputDir.
    
    :param ImageDir: Directory containing input images.
    :param OutputDir: Directory to save results.
    :param Vis: Boolean to save visualized outputs.
    """
    # Load the YOLOv11 segmentation model
    model = YOLO(ModelPath)

    # Run inference on all images first
    results = model(ImageDir, imgsz=1280, save_txt=True, project=OutputDir, 
                    name="Segmentation", save=Vis, iou=0.1, verbose=False)
    
    # Process results
    for result in results:
        if result.masks is not None:  # Check if masks exist in result
            masks = result.masks.data  # Segmentation masks output

            # Extract the first mask
            people_masks = masks[0]
            
            # Convert mask to uint8 and move it to CPU
            people_mask = (people_masks * 255).cpu().numpy().astype(np.uint8)
          
            # Create output directory path for the mask
            mask_dir = Path(OutputDir) / 'Segmentation'/'mask'
            mask_dir.mkdir(parents=True, exist_ok=True)  # Ensure mask directory exists
            
            output_file = mask_dir / os.path.basename(result.path)
            
            # Save the mask as an image
            cv2.imwrite(str(output_file), people_mask)

        else:
            print("No mask for", result.path)
