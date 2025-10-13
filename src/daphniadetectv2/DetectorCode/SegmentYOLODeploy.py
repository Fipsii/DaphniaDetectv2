import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def Segment(ImageDir, OutputDir, ModelPath, Vis=True):
    
    # Load the YOLOv11 segmentation model
    model = YOLO(ModelPath) 
    
    # Run inference on all images first
    results = model(ImageDir, imgsz=1280, save_txt = True,project=OutputDir, name="Segmentation", save = Vis, iou = 0.1, verbose=False)
    # Process results generator
    for result in results:
        masks = result.masks  # Masks object for segmentation masks output
        #print(masks)
       




def Segment_Exp(CroppedDir, ImageDir, OutputDir,ModelPath, Vis=True):
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
    results = model(CroppedDir, imgsz=1280, save_txt=True, project=OutputDir, 
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

    CropSegDir = Path(OutputDir) / 'Segmentation'/ 'labels'
    BoxAnnoDir = Path(OutputDir) / 'Detection'/ 'labels'
    CropImgDir = Path(OutputDir) / 'Detection'/ 'crops' / 'Daphnia'
    AnnotationOutputDir = Path(OutputDir) / 'Segmentation'/ 'labels' 
    OriginalDir = ImageDir
    TranslateDaphniaCropToOriginal(CropSegDir,CropImgDir,BoxAnnoDir,OriginalDir,AnnotationOutputDir,2,False)


def TranslateDaphniaCropToOriginal(CropSegDir, CropImgDir, CropAnnoDir, OriginalDir, OutputDir,
                                   daphnia_class_id=2, visualize=False):
    """
    Translate Daphnia crop segmentation polygons back to original image coordinates
    and save as .txt files. Correctly handles resized crops.

    CropSegDir: directory containing segmentation txt files (base_name_Daphnia.txt)
    CropAnnoDir: directory containing YOLO bounding box annotations of crops (base_name.txt)
    CropImgDir: directory containing resized cropped images (base_name.jpg/png)
    OriginalDir: directory containing original full-size images (base_name.jpg/png)
    OutputDir: directory to save translated YOLO polygon txt files
    daphnia_class_id: class id to use from crop annotations
    visualize: if True, draw polygons on original image
    """
    os.makedirs(OutputDir, exist_ok=True)

    for seg_file in os.listdir(CropSegDir):
        if not seg_file.endswith('_Daphnia.txt'):
            continue

        base_name = seg_file.replace('_Daphnia.txt', '')

        seg_path = os.path.join(CropSegDir, seg_file)
        anno_path = os.path.join(CropAnnoDir, f"{base_name}.txt")

        # --- Read Daphnia bounding box from crop annotation ---
        cls_id = None
        with open(anno_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if int(parts[0]) == daphnia_class_id:
                    cls_id, xc, yc, w, h = map(float, parts[:5])
                    break

        if cls_id is None:
            print(f"Skipping {base_name}: no Daphnia annotation found.")
            continue

        # --- Load resized crop image ---
        crop_file_name = f"{base_name}_Daphnia.jpg"  # or .png depending on your files
        crop_path = os.path.join(CropImgDir, crop_file_name)
        if not os.path.exists(crop_path):
            print(f"Skipping {base_name}: crop image not found.")
            continue
        crop_img = Image.open(crop_path)
        crop_w_resized, crop_h_resized = crop_img.size  # resized crop dimensions

        # --- Load original full image ---
        orig_file_name = f"{base_name}.jpg"  # or .png
        orig_path = os.path.join(OriginalDir, orig_file_name)

        if not os.path.exists(orig_path):
            print(f"Skipping {base_name}: original image not found.")
            continue

        orig_img = Image.open(orig_path).convert("RGB")
        orig_w, orig_h = orig_img.size

        # --- Convert YOLO bbox to original image pixels ---
        x_min = (xc - w/2) * orig_w
        y_min = (yc - h/2) * orig_h
        bbox_w = w * orig_w
        bbox_h = h * orig_h

        # --- Read segmentation polygons ---
        with open(seg_path, 'r') as f:
            seg_lines = f.readlines()

        translated_lines = []
        for line in seg_lines:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            polygon = parts[1:]  # normalized in resized crop [0..1]

            # --- Rescale polygon from resized crop to original bbox ---
            # 1. polygon normalized to resized crop -> scale to original crop size
            x_scale = bbox_w / crop_w_resized
            y_scale = bbox_h / crop_h_resized

            scaled_polygon = []
            for i in range(0, len(polygon), 2):
                # x and y in original image pixels
                x_orig_px = x_min + polygon[i] * crop_w_resized * x_scale
                y_orig_px = y_min + polygon[i+1] * crop_h_resized * y_scale

                # normalized to full image
                x_orig_norm = x_orig_px / orig_w
                y_orig_norm = y_orig_px / orig_h
                scaled_polygon.extend([x_orig_norm, y_orig_norm])

            # Save translated polygon line
            translated_lines.append(f"{cls} " + " ".join(f"{p:.6f}" for p in scaled_polygon) + "\n")

            # Visualization
            if visualize:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(orig_img)
                poly_px = [(scaled_polygon[i]*orig_w, scaled_polygon[i+1]*orig_h)
                           for i in range(0, len(scaled_polygon), 2)]
                draw.polygon(poly_px, outline="red")

        # --- Save translated annotations ---
        out_file = os.path.join(OutputDir, f"{base_name}.txt")
        with open(out_file, 'w') as f:
            f.writelines(translated_lines)

        # Visualization
        if visualize:
            plt.figure(figsize=(8,8))
            plt.imshow(orig_img)
            plt.title(f"Translated Daphnia Polygon: {base_name}")
            plt.axis('off')
            plt.show()