import os
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageOps


def apply_gamma(image_np, gamma=0.5):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image_np, table)


def adaptive_predict_3step(model, raw_img, imgsz=640):
    # 1. Standard
    gray_img = ImageOps.grayscale(raw_img).convert("RGB")
    results_std = model.predict(source=gray_img, conf=0.25, verbose=False, imgsz=imgsz)
    if results_std[0].masks is not None:
        return results_std[0], "1_Standard"

    # 2. Gamma
    gray_np = np.array(ImageOps.grayscale(raw_img))
    gamma_np = apply_gamma(gray_np, gamma=0.5)
    gamma_img = Image.fromarray(gamma_np).convert("RGB")
    
    results_gamma = model.predict(source=gamma_img, conf=0.10, verbose=False, imgsz=imgsz)
    if results_gamma[0].masks is not None:
        return results_gamma[0], "2_Gamma_Ghost"

    # 3. CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced_np = clahe.apply(gray_np)
    clahe_img = Image.fromarray(enhanced_np).convert("RGB")
    
    results_clahe = model.predict(source=clahe_img, conf=0.15, verbose=False, imgsz=imgsz)
    return results_clahe[0], "3_Enhanced_CLAHE"

# --- Main Function ---
def Segment_Exp(ImageDir, OutputDir, ModelPath, Vis=True):
    #print(f"Loading model from: {ModelPath}")
    model = YOLO(ModelPath)
    
    # Setup Directories
    mask_output_dir = Path(OutputDir) / 'Segmentation' / 'mask'
    label_output_dir = Path(OutputDir) / 'Segmentation' / 'labels'  
    vis_output_dir = Path(OutputDir) / 'Segmentation' / 'vis'
    
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)            
    if Vis:
        vis_output_dir.mkdir(parents=True, exist_ok=True)

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    image_files = [f for f in os.listdir(ImageDir) if f.lower().endswith(valid_extensions)]
    
    #print(f"Found {len(image_files)} images. Starting adaptive inference...")

    for i, img_filename in enumerate(image_files):
        try:
            img_path = os.path.join(ImageDir, img_filename)
            raw_img = Image.open(img_path)
            
            # Run Adaptive Prediction
            result, method_used = adaptive_predict_3step(model, raw_img, imgsz=640) # Adjust imgsz if needed
            
            if result.masks is not None:
                # --- A. Save Binary Mask (Image) ---
                masks_data = result.masks.data
                people_mask = (masks_data[0] * 255).cpu().numpy().astype(np.uint8)
                
                # Resize if YOLO output size != Original size
                if people_mask.shape != raw_img.size[::-1]:
                    people_mask = cv2.resize(people_mask, raw_img.size, interpolation=cv2.INTER_NEAREST)

                cv2.imwrite(str(mask_output_dir / img_filename), people_mask)

                # --- B. Save YOLO Label (.txt) --- 
                # Format: class x1 y1 x2 y2 ... (Normalized)
                label_path = label_output_dir / f"{Path(img_filename).stem}.txt"
                
                with open(label_path, 'w') as f:
                    # Iterate over all detected objects in this image
                    # result.masks.xyn gives normalized polygon coordinates
                    for j, seg in enumerate(result.masks.xyn):
                        cls = int(result.boxes.cls[j].item())
                        # Flatten coordinates to string: "x1 y1 x2 y2..."
                        coords = " ".join([f"{coord:.6f}" for coord in seg.flatten()])
                        f.write(f"{cls} {coords}\n")

                # --- C. Save Visualization ---
                if Vis:
                    annotated_frame = result.plot()
                    # Color Label
                    color = (0, 255, 0)
                    if "Gamma" in method_used: color = (0, 0, 255)
                    elif "CLAHE" in method_used: color = (0, 165, 255)
                    
                    cv2.putText(annotated_frame, f"Mode: {method_used}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    cv2.imwrite(str(vis_output_dir / f"pred_{img_filename}"), annotated_frame)
                    
                #print(f"[{i+1}/{len(image_files)}] {img_filename} -> {method_used} (Labels Saved)")
            
            else:
                print(f"[{i+1}/{len(image_files)}] {img_filename} -> NO OBJECT FOUND")

        except Exception as e:
            print(f"Error processing {img_filename}: {e}")

    print(f"Done. Outputs in: {OutputDir}")


