from ultralytics import YOLO
import time
import os
import torch
import os
import cv2
import numpy as np
import glob
import pandas as pd
import torch
import sahi
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from PIL import Image

def Images_list(path_to_images):
    # Check if the input is already a list of paths
    if isinstance(path_to_images, list):
        image_list = path_to_images
        image_names = [os.path.basename(p) for p in path_to_images]
        return image_list, image_names

    # Original logic for directory strings
    image_list = []
    image_names = []
    for root, dirs, files in os.walk(path_to_images, topdown=False):
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                image_list.append(os.path.join(root, name))
                image_names.append(name)
    return image_list, image_names
    
def CropImagesFromYOLO(Original_Images, labels_folder, Crop_mode, Save_folder, class_mapping):
    """
    Crops images based on YOLO format annotations and saves them.
    
    Parameters:
    - Original_Images: List of image paths
    - YOLO_Annotations: List of YOLO annotation file paths (corresponding to images)
    - Crop_mode: List of class names to crop (e.g., ["Body", "Eye"])
    - Save_folder: Folder where cropped images will be saved
    - class_mapping: Dictionary mapping class names to YOLO class IDs (e.g., {"Body": 0, "Eye": 1})
    
    Returns:
    - Dictionary with class names as keys and lists of cropped images as values
    """
    Original_Images, Image_names = Images_list(Original_Images)

    os.makedirs(Save_folder, exist_ok=True)
    cropped_images = {cls: [] for cls in Crop_mode}
    YOLO_Annotations = glob.glob(os.path.join(labels_folder, "*.txt"))

    for img_path, img_name in zip(Original_Images, Image_names):
        try:

            # Get the base name of the image (without extension)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            

            # Find the corresponding annotation file by replacing the extension
            ann_path = None
            for ann in YOLO_Annotations:
                if base_name == os.path.splitext(os.path.basename(ann))[0]:
                    ann_path = ann
                    break
            
            if ann_path is None:
                print(f"Annotation file for {img_path} not found.")
                continue
                

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image {img_path}")
                continue

            height, width = img.shape[:2]  # Get image dimensions

            with open(ann_path, "r") as f:
                lines = f.readlines()  # Read YOLO annotations
            
            for line in lines:
                values = line.strip().split()
                

                class_id = int(float(values[0]))  # YOLO class ID
                
                # Convert class ID to class name
                class_name = class_mapping.get(class_id, None)  # Directly get the class name from the dictionary
                
                if class_name not in Crop_mode:
                    continue  # Skip classes not in Crop_mode
                
                # Convert YOLO normalized coordinates to absolute pixel coordinates
                # The values 6 is for some reason never  float but int in annotations
                
                x_center, y_center, bbox_width, bbox_height = map(float, values[1:])

                # Calculate absolute pixel coordinates
                Xmin = int((x_center - bbox_width / 2) * width)
                Ymin = int((y_center - bbox_height / 2) * height)
                Xmax = int((x_center + bbox_width / 2) * width)
                Ymax = int((y_center + bbox_height / 2) * height)
                
                # Ensure coordinates are within image bounds (avoid negative or too large values)
                Xmin = max(0, Xmin)
                Ymin = max(0, Ymin)
                Xmax = min(width, Xmax)
                Ymax = min(height, Ymax)
		
		
                # Crop the image
                crop = img[Ymin:Ymax, Xmin:Xmax]
                debug_img = img.copy()
                #plt.imshow(debug_img)
                #plt.plot([Xmin, Xmax, Xmax, Xmin, Xmin], [Ymin, Ymin, Ymax, Ymax, Ymin], color="red", linewidth=2)
                #plt.show()
                if crop.size == 0:
                    print(f"Invalid crop for {img_name}, class {class_name}")
                    continue
                
                # Save cropped image
                
                class_folder = os.path.join(Save_folder, class_name)
                os.makedirs(class_folder, exist_ok=True)
                
                save_path = os.path.join(class_folder, f"{os.path.splitext(os.path.basename(img_path))[0]}_{class_name}.jpg")

                cv2.imwrite(save_path, crop)
                
                # Store the crop in the dictionary
                cropped_images[class_name].append(crop)

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return cropped_images




def DetectOrgans(Images, OutputDir, vis=True, NMS=True, refineTip=True, organs=["Daphnia", "Spina base"], ModelPath=None, SpinaModelPath=None, conf=0.01, iou=0.2, use_sahi=False, slice_size=1280, overlap=0.2):
    '''
    Detect specified organs with optional Sliced Aided Hyper Inference (SAHI).
    Geometric filtering applied to Spina Tips (7) followed by NMS.
    '''
    obj_detect_start = time.time()
    
    os.makedirs(os.path.join(OutputDir, "Detection", "labels"), exist_ok=True)
    detection_data = []

    # Format Images input to a list of paths for consistent iteration
    if isinstance(Images, str) and os.path.isdir(Images):
        image_paths = [os.path.join(Images, f) for f in os.listdir(Images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    elif isinstance(Images, str):
        image_paths = [Images]
    else:
        image_paths = Images

    # Initialize Models
    if use_sahi:

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="yolov11",
            model_path=ModelPath,
            confidence_threshold=conf,
            device=device
        )
        class_names = {int(k): v for k, v in sahi_model.category_mapping.items()}
    else:
        model = YOLO(ModelPath)
        class_names = model.names

    for img_path in image_paths:
        image_name = os.path.basename(img_path)
        combined = None

        if use_sahi:
            # SAHI Inference Pass
            with Image.open(img_path) as img:
                w_img, h_img = img.size
                
            result = get_sliced_prediction(
                img_path,
                sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                postprocess_type="NMM"
            )


            rows = []
            for pred in result.object_prediction_list:
                x_min, y_min, x_max, y_max = pred.bbox.to_xyxy()
                x_c = ((x_min + x_max) / 2.0) / w_img
                y_c = ((y_min + y_max) / 2.0) / h_img
                w_norm = (x_max - x_min) / w_img
                h_norm = (y_max - y_min) / h_img
                rows.append([pred.category.id, x_c, y_c, w_norm, h_norm, pred.score.value])
            
            if rows:
                combined = torch.tensor(rows, dtype=torch.float32)
            else:
                final_tensor = torch.empty((0, 5))
                
        else:
            # Standard YOLO Inference Pass
            results = model(img_path, imgsz=1280, conf=conf, iou=iou, verbose=False)
            result = results[0]
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                final_tensor = torch.empty((0, 5))
            else:
                classes = boxes.cls.unsqueeze(1)
                conf_scores = boxes.conf.unsqueeze(1)
                combined = torch.cat((classes, boxes.xywhn, conf_scores), dim=1)

        # Filtering Logic
        if combined is not None and len(combined) > 0:
            if NMS:
                other_organs = combined[combined[:, 0] != 7]
                final_rows = []
                seen_classes = set()
                
                other_organs = other_organs[torch.argsort(other_organs[:, 5], descending=True)]
                
                for row in other_organs:
                    cid = int(row[0].item())
                    if cid not in seen_classes:
                        final_rows.append(row)
                        seen_classes.add(cid)
                
                tip_candidates = combined[combined[:, 0] == 7]
                
                eye_row = next((r for r in final_rows if r[0] == 3), None) 
                base_row = next((r for r in final_rows if r[0] == 6), None) 
                
                valid_tips_list = []
                if eye_row is not None and base_row is not None and tip_candidates.size(0) > 0:
                    p_eye, p_base = eye_row[1:3], base_row[1:3]
                    v_base_eye = p_eye - p_base
                    
                    for tip in tip_candidates:
                        v_base_tip = tip[1:3] - p_base
                        norm_prod = torch.norm(v_base_eye) * torch.norm(v_base_tip)
                        
                        if norm_prod > 1e-8:
                            cos_theta = torch.dot(v_base_eye, v_base_tip) / norm_prod
                            angle_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
                            if angle_rad >= 2.0: 
                                valid_tips_list.append(tip)
                else:
                    valid_tips_list = [t for t in tip_candidates]

                if valid_tips_list:
                    valid_tips_tensor = torch.stack(valid_tips_list)
                    best_idx = torch.argmax(valid_tips_tensor[:, 5])
                    final_rows.append(valid_tips_tensor[best_idx])
                   
                for row in final_rows:
                    detection_data.append({
                        "Image": image_name,
                        "Class": class_names[int(row[0].item())],
                        "Confidence": float(row[5].item())
                    })

                final_tensor = torch.stack(final_rows)[:, :5] if final_rows else torch.empty((0, 5))

            else:
                final_tensor = combined[:, :5]
                for row in combined:
                    detection_data.append({
                        "Image": image_name,
                        "Class": class_names[int(row[0].item())],
                        "Confidence": float(row[5].item())
                    })

        # Save to file
        label_saveloc = os.path.join(OutputDir, "Detection", "labels", f"{os.path.splitext(image_name)[0]}.txt")
        with open(label_saveloc, "w") as f:
            for row in final_tensor:
                f.write(" ".join(map(str, row.tolist())) + "\n")
    
    # --- POST-PROCESSING STEPS ---
    labels_folder = os.path.join(OutputDir, "Detection", "labels")

    if refineTip:
        print("\nDetection of missing spina tips with specialised model")
        SpinaTipEnhance(image_paths, labels_folder, SpinaModelPath)
        
    for filename in os.listdir(labels_folder):
        file_path = os.path.join(labels_folder, filename)
        update_daphnid_bounding_boxes(file_path)

    print("\nDetection finished. Cropping images now...")
    CropImagesFromYOLO(
        image_paths,
        labels_folder=labels_folder,
        Crop_mode=organs, 
        Save_folder=os.path.join(OutputDir, "Detection", "crops"),
        class_mapping=class_names
    )
        
    if vis:
        print("\nDrawing detection boxes...")
        DrawYOLOBoxes(
            Original_Images=image_paths,
            labels_folder=labels_folder,
            Save_folder=os.path.join(OutputDir, "Detection", "visuals"),
            class_mapping=class_names
        )
    
    print(f"\nAnnotations saved to: {OutputDir}")
    df = pd.DataFrame(detection_data)

    return OutputDir, df
    
    
def update_daphnid_bounding_boxes(annotation_file):
    if not os.path.exists(annotation_file):
        print(f"Annotation file {annotation_file} does not exist.")
        return

    # Read all annotations
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    # Initialize enclosing coordinates
    x1_min, y1_min = float('inf'), float('inf')
    x2_max, y2_max = float('-inf'), float('-inf')
    has_daphnia = False

    # Convert all boxes to corners and find overall bounding box
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        class_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])

        # YOLO coords -> corners
        x1 = xc - w/2
        y1 = yc - h/2
        x2 = xc + w/2
        y2 = yc + h/2

        boxes.append((class_id, x1, y1, x2, y2))

        # Keep track if there's a Daphnia body
        if class_id == 2:
            has_daphnia = True

        # Update enclosing box over all boxes
        x1_min = min(x1_min, x1)
        y1_min = min(y1_min, y1)
        x2_max = max(x2_max, x2)
        y2_max = max(y2_max, y2)

    if not has_daphnia:
        print(f"No Daphnid body found in {annotation_file}")
        return

    # Convert back to YOLO format
    new_xc = (x1_min + x2_max) / 2
    new_yc = (y1_min + y2_max) / 2
    new_w = x2_max - x1_min
    new_h = y2_max - y1_min

    # Update only Daphnia body boxes
    updated_lines = []
    for class_id, x1, y1, x2, y2 in boxes:
        if class_id == 2:
            updated_lines.append(f"{class_id} {new_xc:.6f} {new_yc:.6f} {new_w:.6f} {new_h:.6f}\n")
        else:
            # Keep other boxes as is
            xc = (x1 + x2) / 2
            yc = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            updated_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    with open(annotation_file, 'w') as f:
        f.writelines(updated_lines)
        
        
def DrawYOLOBoxes(Original_Images, labels_folder, Save_folder, class_mapping=None):
    """
    Draws YOLO format bounding boxes on images with distinct colors per class,
    adds detection probabilities, and marks midpoints with a small crosshair
    for Spina tip, Spina base, and Eye.
    """
    import os, glob, cv2

    # Nicer distinct colors (BGR)
    fixed_colors = {
        'Body': (34, 139, 34),        # Forest Green
        'Brood cavity': (70, 130, 180),# Steel Blue
        'Daphnia': (255, 215, 0),     # Gold
        'Eye': (220, 20, 60),         # Crimson
        'Head': (0, 191, 255),        # Deep Sky Blue
        'Heart': (255, 20, 147),      # Deep Pink
        'Spina base': (138, 43, 226), # Blue Violet
        'Spina tip': (0, 206, 209),   # Dark Turquoise
        'SpinaTipBase': (255, 140, 0) # Dark Orange
    }

    crosshair_classes = ['Spina tip', 'Spina base', 'Eye']

    # Handle folder or list input
    if isinstance(Original_Images, str):
        Original_Images = glob.glob(os.path.join(Original_Images, "*.*"))

    Image_names = [os.path.basename(p) for p in Original_Images]
    os.makedirs(Save_folder, exist_ok=True)
    YOLO_Annotations = glob.glob(os.path.join(labels_folder, "*.txt"))

    for img_path, img_name in zip(Original_Images, Image_names):
        try:
            base_name = os.path.splitext(img_name)[0]

            # Find corresponding annotation file
            ann_path = next((ann for ann in YOLO_Annotations
                             if base_name == os.path.splitext(os.path.basename(ann))[0]), None)
            if ann_path is None:
                print(f"Annotation file for {img_path} not found.")
                continue

            img = cv2.imread(img_path)
            if img is None:
                print(f"Error reading image {img_path}")
                continue

            height, width = img.shape[:2]

            with open(ann_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                values = line.strip().split()
                if len(values) < 5:
                    continue

                class_id = int(float(values[0]))
                class_name = class_mapping.get(class_id, str(class_id)) if class_mapping else str(class_id)

                # YOLO coords to absolute pixel coords
                x_center, y_center, bbox_width, bbox_height = map(float, values[1:5])
                Xmin = int((x_center - bbox_width / 2) * width)
                Ymin = int((y_center - bbox_height / 2) * height)
                Xmax = int((x_center + bbox_width / 2) * width)
                Ymax = int((y_center + bbox_height / 2) * height)

                Xmin, Ymin = max(0, Xmin), max(0, Ymin)
                Xmax, Ymax = min(width, Xmax), min(height, Ymax)

                # Fixed color per class
                color = fixed_colors.get(class_name, (255, 255, 255))

                # Draw rectangle
                cv2.rectangle(img, (Xmin, Ymin), (Xmax, Ymax), color, 2)

                # Draw midpoint crosshair for specific classes
                if class_name in crosshair_classes:
                    cx, cy = (Xmin + Xmax) // 2, (Ymin + Ymax) // 2
                    cross_size = 5  # pixels
                    cv2.line(img, (cx - cross_size, cy), (cx + cross_size, cy), color, 1)
                    cv2.line(img, (cx, cy - cross_size), (cx, cy + cross_size), color, 1)

                # Label text
                cv2.putText(img, class_name, (Xmin, max(0, Ymin - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

            # Save annotated image
            save_path = os.path.join(Save_folder, f"{base_name}_boxed.jpg")
            cv2.imwrite(save_path, img)


        except Exception as e:
            print(f"Error processing {img_path}: {e}")





import os
import cv2
import numpy as np
from ultralytics import YOLO

def RedetectSpinaTipYOLO(image_path, model, spina_base, eyes, target_imgsz=1280):
    """
    Uses the exact Eye-Base rotation logic to crop a high-resolution ROI 
    for YOLO redetection.
    """
    img = cv2.imread(image_path)
    if img is None: return {'detections': []}
    h, w = img.shape[:2]

    # 1. Parse Geometry (same as your U-Net logic)
    x_center, y_center, width_frac = spina_base
    x_center_pix, y_center_pix = int(x_center * w), int(y_center * h)
    eye_x_avg = np.mean([x for x, y in eyes]) * w
    eye_y_avg = np.mean([y for x, y in eyes]) * h

    # 2. Rotation: Align Eye-Base axis to Vertical
    dx, dy = eye_x_avg - x_center_pix, eye_y_avg - y_center_pix
    angle = np.degrees(np.arctan2(dy, dx))
    
    # We add 90 to make the Eye -> Base vector point "down" (positive Y)
    M_rot = cv2.getRotationMatrix2D((x_center_pix, y_center_pix), angle + 90, 1.0)
    rotated_img = cv2.warpAffine(img, M_rot, (w, h))

    # 3. Handle Flip: Ensure Eye is 'above' the Base in the rotated frame
    eye_coords = np.array([[[eye_x_avg, eye_y_avg]]], dtype=np.float32)
    eye_rotated = cv2.transform(eye_coords, M_rot)[0][0]
    
    did_flip = False
    # If eye is below base in rotated coordinates, flip vertically
    if eye_rotated[1] > y_center_pix:
        rotated_img = cv2.flip(rotated_img, 0)
        y_center_rot = h - y_center_pix
        did_flip = True
    else:
        y_center_rot = y_center_pix

    # 4. Define ROI (Crops 'downward' from the base)
    half_width = int((width_frac * 2.0) * w / 2) # Wider margin for safety
    roi_length = int(0.4 * h) # 40% of image height
    
    y_start = int(y_center_rot)
    y_end = int(min(h, y_start + roi_length))
    x_start = int(max(0, x_center_pix - half_width))
    x_end = int(min(w, x_center_pix + half_width))

    roi = rotated_img[y_start:y_end, x_start:x_end]
    if roi.size == 0: return {'detections': []}

    # 5. YOLO Inference on the oriented ROI
    results = model(roi, imgsz=target_imgsz, conf=0.001, verbose=False)[0]
    
    extra_detections = []
    for box in results.boxes:
        if int(box.cls[0]) == 7: # Spina Tip
            # Local ROI -> Rotated Image
            rx1, ry1, rx2, ry2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            
            mid_x = (rx1 + rx2) / 2 + x_start
            mid_y = (ry1 + ry2) / 2 + y_start
            
            # Undo Flip
            if did_flip:
                mid_y = h - mid_y
            
            # Undo Rotation (Map back to Original Image)
            M_inv = cv2.invertAffineTransform(M_rot)
            orig_pt = cv2.transform(np.array([[[mid_x, mid_y]]], dtype=np.float32), M_inv)[0][0]
            
            # Synthetic 2% box
            bw, bh = 0.02 * w, 0.02 * h
            extra_detections.append((7, orig_pt[0]-bw/2, orig_pt[1]-bh/2, orig_pt[0]+bw/2, orig_pt[1]+bh/2, conf))

    # Return only the highest confidence tip
    if extra_detections:
        extra_detections = [max(extra_detections, key=lambda x: x[5])]

    return {'detections': extra_detections}
    
    
    
def SpinaTipEnhance(image_dir, label_dir, model_path):
    model = YOLO(model_path)
    valid_exts = (".jpg", ".jpeg", ".png")
    
    for img_name in [f for f in os.listdir(image_dir) if f.lower().endswith(valid_exts)]:
        img_path = os.path.join(image_dir, img_name)
        lbl_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")
        
        if not os.path.exists(lbl_path): continue

        # Load existing labels
        with open(lbl_path, "r") as f:
            lines = [line.strip().split() for line in f.readlines()]
        
        # Convert to float first to handle strings like '0.0', then cast to int
        labels = [(int(float(l[0])), float(l[1]), float(l[2]), float(l[3]), float(l[4])) for l in lines]
        
        base = next(((l[1], l[2], l[3]) for l in labels if l[0] == 6), None)
        eyes = [(l[1], l[2]) for l in labels if l[0] == 3]
        has_tip = any(l[0] == 7 for l in labels)

        if base and eyes and not has_tip:
            print(f"Scanning ROI for missed tip: {img_name}")
            res = RedetectSpinaTipYOLO(img_path, model, base, eyes)
            
            if res['detections']:
                h, w = cv2.imread(img_path).shape[:2]
                with open(lbl_path, "a") as f:
                    for det in res['detections']:
                        cls, x1, y1, x2, y2, _ = det
                        # Convert back to YOLO normalized format
                        cx, cy = ((x1+x2)/2)/w, ((y1+y2)/2)/h
                        bw, bh = (x2-x1)/w, (y2-y1)/h
                        f.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                print(" -> Success: Tip detected in ROI and added.")
                
                
                

