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

def adaptive_predict_4step(model, raw_img, imgsz=640):
    # Base grayscale array
    gray_np = np.array(ImageOps.grayscale(raw_img))
    h, w = gray_np.shape
    pad = int(max(h, w) * 0.1)

    def apply_padding(img_array):
        return cv2.copyMakeBorder(img_array, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)

    # 1. Padded Standard
    padded_base_np = apply_padding(gray_np)
    padded_base_img = Image.fromarray(padded_base_np).convert("RGB")
    res = model.predict(source=padded_base_img, conf=0.10, verbose=False, imgsz=imgsz)[0]
    if res.masks is not None:
        return res, "1_Padded_Standard", pad

    # 2. Padded Gamma
    # Assumes apply_gamma(arr, gamma) is defined in your scope
    gamma_np = apply_gamma(gray_np, gamma=0.5) 
    padded_gamma_np = apply_padding(gamma_np)
    padded_gamma_img = Image.fromarray(padded_gamma_np).convert("RGB")
    res = model.predict(source=padded_gamma_img, conf=0.10, verbose=False, imgsz=imgsz)[0]
    if res.masks is not None:
        return res, "2_Padded_Gamma", pad

    # 3. Padded CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    clahe_np = clahe.apply(gray_np)
    padded_clahe_np = apply_padding(clahe_np)
    padded_clahe_img = Image.fromarray(padded_clahe_np).convert("RGB")
    res = model.predict(source=padded_clahe_img, conf=0.10, verbose=False, imgsz=imgsz)[0]
    if res.masks is not None:
        return res, "3_Padded_CLAHE", pad

    # 4. Standard No Pad (Fallback)
    base_img = Image.fromarray(gray_np).convert("RGB")
    res = model.predict(source=base_img, conf=0.10, verbose=False, imgsz=imgsz)[0]
    
    return res, "4_Standard_No_Pad", 0


def Segment_Exp(ImageDir, OutputDir, ModelPath, Vis=True):
    model = YOLO(ModelPath)
    out_path = Path(OutputDir)
    crop_dir = out_path / 'Detection' / 'crops' / 'Daphnia'
    yolo_label_dir = out_path / 'Detection' / 'labels'
    
    mask_output_dir = out_path / 'Segmentation' / 'mask'
    label_output_dir = out_path / 'Segmentation' / 'labels'  
    vis_output_dir = out_path / 'Segmentation' / 'vis'
    
    mask_output_dir.mkdir(parents=True, exist_ok=True)
    label_output_dir.mkdir(parents=True, exist_ok=True)            
    if Vis:
        vis_output_dir.mkdir(parents=True, exist_ok=True)

    if not crop_dir.exists() or not yolo_label_dir.exists():
        print("Error: Required directories missing.")
        return

    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
    crop_files = [f for f in os.listdir(crop_dir) if f.lower().endswith(valid_extensions)]

    for crop_filename in crop_files:
        try:
            base_stem = crop_filename.rsplit('_Daphnia', 1)[0]
            orig_filename = next((f for f in os.listdir(ImageDir) if f.startswith(base_stem) and f.lower().endswith(valid_extensions)), None)
            
            if not orig_filename: 
                continue
                
            orig_path = os.path.join(ImageDir, orig_filename)
            crop_path = os.path.join(crop_dir, crop_filename)
            label_path = yolo_label_dir / f"{base_stem}.txt"
            
            if not label_path.exists(): 
                continue

            orig_img = Image.open(orig_path)
            raw_crop = Image.open(crop_path)
            orig_w, orig_h = orig_img.size
            crop_w, crop_h = raw_crop.size
            
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            daphnia_line = next((line for line in lines if line.startswith('2 ')), None)
            if not daphnia_line: 
                continue
                
            _, x_c, y_c, w, h = map(float, daphnia_line.strip().split())
            x_min = max(0, int((x_c - w / 2) * orig_w))
            y_min = max(0, int((y_c - h / 2) * orig_h))

            # Execute adaptive inference
            result, method_used, pad = adaptive_predict_4step(model, raw_crop, imgsz=640)
            
            if result.masks is not None:
                pred_h, pred_w = (crop_h + 2 * pad, crop_w + 2 * pad)

                # 1. Process Pixel Mask
                masks_data = result.masks.data.cpu().numpy()
                raw_m = (masks_data[0] * 255).astype(np.uint8)
                
                if raw_m.shape != (pred_h, pred_w):
                    raw_m = cv2.resize(raw_m, (pred_w, pred_h), interpolation=cv2.INTER_NEAREST)

                crop_mask = raw_m[pad : pad + crop_h, pad : pad + crop_w] if pad > 0 else raw_m

                full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
                h_end, w_end = min(orig_h, y_min + crop_h), min(orig_w, x_min + crop_w)
                full_mask[y_min:h_end, x_min:w_end] = crop_mask[:h_end-y_min, :w_end-x_min]
                cv2.imwrite(str(mask_output_dir / orig_filename), full_mask)

                # 2. Process YOLO Labels
                seg_label_path = label_output_dir / f"{base_stem}.txt"
                with open(seg_label_path, 'w') as f_out:
                    for j, seg in enumerate(result.masks.xyn):
                        cls = int(result.boxes.cls[j].item())
                        orig_seg = []
                        for point in seg:
                            abs_x = (point[0] * pred_w) - pad + x_min
                            abs_y = (point[1] * pred_h) - pad + y_min
                            orig_seg.append([abs_x / orig_w, abs_y / orig_h])
                        
                        coords = " ".join([f"{coord:.6f}" for coord in np.array(orig_seg).flatten()])
                        f_out.write(f"{cls} {coords}\n")

                if Vis:
                    annotated_crop = result.plot()
                    if pad > 0: 
                        annotated_crop = annotated_crop[pad:pad+crop_h, pad:pad+crop_w]
                    
                    orig_cv2 = cv2.cvtColor(np.array(orig_img), cv2.COLOR_RGB2BGR)
                    orig_cv2[y_min:h_end, x_min:w_end] = annotated_crop[:h_end-y_min, :w_end-x_min]
                    
                    color = (0, 255, 0)
                    if "Gamma" in method_used: color = (0, 0, 255)
                    elif "CLAHE" in method_used: color = (0, 165, 255)
                    
                    cv2.putText(orig_cv2, f"Mode: {method_used}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    cv2.imwrite(str(vis_output_dir / f"pred_{orig_filename}"), orig_cv2)

        except Exception as e:
            print(f"Error processing {crop_filename}: {e}")


