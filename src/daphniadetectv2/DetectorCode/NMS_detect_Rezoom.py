from ultralytics import YOLO
import time
import os
import torch
import os
import cv2
import numpy as np
import glob

def Images_list(path_to_images):
  ## Takes path, creates list of image names and full paths for all
  ## PNGS or JPGS in the folder
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    #print(dirs, files)
    for name in files:
      _, ext = os.path.splitext(name)
      if ext.lower() in ['.jpg', '.jpeg', '.png'] and name != '.DS_Store':
        #print(os.path.join(root, name))
        Image_names.append(os.path.join(root, name))
        PureNames.append(name)
        #print(files)
  return Image_names, PureNames

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

import torch
import time
import os
from ultralytics import YOLO

def DetectOrgans(Images, OutputDir, vis=True, NMS=True, crop=False, refineTip=True, organs=["Eye"], ModelPath=None, SpinaModelPath=None, conf=0.01, iou=0.2):
    '''
    Detect specified organs in a list of images using a YOLO model and save the detection results.
    '''

    obj_detect_start = time.time()

    # Initialize YOLO model
    model = YOLO(ModelPath)

    # Run YOLO model
    # Note: Stream=True returns a generator
    results = model(Images, stream=True, imgsz=1280, conf=conf, iou=iou, project=OutputDir, name="Detection", verbose=False)
    obj_detect_end = time.time()
    
    # Create storage folder:
    os.makedirs(OutputDir + "/Detection/labels", exist_ok=True)

    # --- MAIN LOOP FIX ---
    for result in results:
        # Remove the nested "for result in results:" loop here
        boxes = result.boxes
        
        # Safe check for empty detections
        if boxes is None or len(boxes) == 0:
            final_tensor = torch.empty((0, 5))
        else:
            classes = boxes.cls.unsqueeze(1)
            conf_scores = boxes.conf.unsqueeze(1)
            
            # Create combined tensor for NMS processing: [class, x, y, w, h, conf]
            combined = torch.cat((classes, boxes.xywhn, conf_scores), dim=1)

            if NMS:
                # 1. Standard NMS for all organs EXCEPT Spina Tip (7)
                other_organs = combined[combined[:, 0] != 7]
                final_rows = []
                seen_classes = set()
                
                # Sort by confidence descending
                other_organs = other_organs[torch.argsort(other_organs[:, 5], descending=True)]
                
                for row in other_organs:
                    cid = row[0].item()
                    if cid not in seen_classes:
                        final_rows.append(row)
                        seen_classes.add(cid)
                
                # 2. Geometric Filtering for Spina Tips (7)
                tip_candidates = combined[combined[:, 0] == 7]
                
                # Get reference points from filtered organs
                eye_row = next((r for r in final_rows if r[0] == 3), None)
                base_row = next((r for r in final_rows if r[0] == 6), None)
                
                if eye_row is not None and base_row is not None and tip_candidates.size(0) > 0:
                    p_eye = eye_row[1:3]
                    p_base = base_row[1:3]
                    v_base_eye = p_eye - p_base
                    
                    valid_tips = []
                    for tip in tip_candidates:
                        p_tip = tip[1:3]
                        v_base_tip = p_tip - p_base
                        
                        # Angle calculation via dot product
                        cos_theta = torch.dot(v_base_eye, v_base_tip) / (torch.norm(v_base_eye) * torch.norm(v_base_tip))
                        angle_rad = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

                        if angle_rad >= 2.618: #150° - 210° (approx 2.618 rads)
                            valid_tips.append(tip)
                    
                    # 3. NMS for Tips: Keep the highest confidence valid tip
                    if valid_tips:
                        valid_tips = torch.stack(valid_tips)
                        best_tip = valid_tips[torch.argmax(valid_tips[:, 5])]
                        final_rows.append(best_tip)
                elif tip_candidates.size(0) > 0:
                    # Fallback logic if needed, currently pass
                    pass

                # Stack rows and remove confidence column to match standard YOLO format (5 cols)
                # combined shape was 6 cols (last is conf). We slice [:, :-1]
                if final_rows:
                    final_tensor = torch.stack(final_rows)[:, :-1]
                else:
                    final_tensor = torch.empty((0, 5))

            else:
                # --- ERROR FIX ---
                # Use boxes.xywhn instead of boxes
                # classes is already unsqueezed at the top
                final_tensor = torch.cat((classes, boxes.xywhn), dim=1) # Shape: (N, 5)

        # Save to file
        label_saveloc = f"{OutputDir}/Detection/labels/{os.path.splitext(os.path.basename(result.path))[0]}.txt"
        with open(label_saveloc, "w") as f:
            for row in final_tensor:
                f.write(" ".join(map(str, row.tolist())) + "\n")
    
    # Post-processing steps
    labels_folder = os.path.join(OutputDir, "Detection", "labels")
    # Assuming update_daphnid_bounding_boxes is defined elsewhere
    # for filename in os.listdir(labels_folder):
    #     file_path = os.path.join(labels_folder, filename)
    #     update_daphnid_bounding_boxes(file_path)

    if refineTip:
        print("\nDetection of missing spina tips with specialised model")
        # Ensure SpinaTipEnhance is imported or defined
        SpinaTipEnhance(Images, labels_folder, SpinaModelPath)
   
    if crop:
       print("\nDetection finished. Cropping images now...")
       CropImagesFromYOLO(
        Images,
        labels_folder=labels_folder,
        Crop_mode=organs,
        Save_folder=os.path.join(OutputDir, "Detection", "crops"),
        class_mapping=result.names
       )
        
    if vis:
        print("\nDrawing detection boxes...")
        DrawYOLOBoxes(
            Original_Images=Images,
            labels_folder=labels_folder,
            Save_folder=os.path.join(OutputDir, "Detection", "visuals"),
            class_mapping=result.names
        )
    
    print(f"\nAnnotations saved to: {OutputDir}")
    print("Please review the results for any potential issues.\n")

    return OutputDir

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
        'Spina tip': (0, 206, 209)    # Dark Turquoise
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



def resize_with_zoom(img, target_size):
    h, w = img.shape[:2]

    # Avoid empty image
    if h == 0 or w == 0:
        raise ValueError("Cannot resize empty image (height or width is zero).")

    # Scale factor to make the largest dimension = target_size
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize only
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    return resized, scale, 0, 0  # left/top = 0 since no padding




## Test

import os
import cv2
import torch
import torch.nn as nn
import numpy as np

# --- 1. U-Net Architecture ---
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SpineTipDualUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = DoubleConv(1, 64); self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(64, 128)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        self.out = nn.Conv2d(64, 2, 1) 

    def forward(self, x):
        c1 = self.enc1(x); c2 = self.enc2(self.pool1(c1))
        bn = self.bottleneck(self.pool1(c2))
        u2 = torch.cat([self.up2(bn), c2], dim=1); u2 = self.dec2(u2)
        u1 = torch.cat([self.up1(u2), c1], dim=1); u1 = self.dec1(u1)
        return self.out(u1) 

# --- 2. Preprocessing Helper ---
def letterbox(img, new_shape=(128, 128)):
    h, w = img.shape[:2]
    r = min(new_shape[1] / h, new_shape[0] / w)
    nw, nh = int(w * r), int(h * r)
    resized = cv2.resize(img, (nw, nh))
    
    pad_val = int(np.mean(resized))
    canvas = np.full((new_shape[1], new_shape[0]), pad_val, dtype=np.uint8)
    dw, dh = (new_shape[0] - nw) // 2, (new_shape[1] - nh) // 2
    canvas[dh:dh+nh, dw:dw+nw] = resized
    return canvas, r, dw, dh

# --- 3. Core Detection (Replaces YOLO) ---
def RedetectSpinaTip(image_path, model_path, spina_base, eyes, model_input_size=(128, 128)):
    """
    Uses the Dual U-Net to find the tip coordinate, maps it back to the original 
    image geometry, and creates a synthetic bounding box for YOLO formatting.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SpineTipDualUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img = cv2.imread(image_path)
    if img is None: raise FileNotFoundError(f"Image not found: {image_path}")
    h, w, _ = img.shape

    x_center, y_center, width_frac = spina_base
    x_center_pix, y_center_pix = int(x_center * w), int(y_center * h)
    eye_x_avg = np.mean([x for x, y in eyes]) * w
    eye_y_avg = np.mean([y for x, y in eyes]) * h

    # Rotation
    dx, dy = eye_x_avg - x_center_pix, eye_y_avg - y_center_pix
    angle = np.degrees(np.arctan2(dy, dx))
    M_rot = cv2.getRotationMatrix2D((x_center_pix, y_center_pix), angle + 90, 1.0)
    rotated_img = cv2.warpAffine(img, M_rot, (w, h))

    # Black border crop
    mask = rotated_img.sum(axis=2) > 0
    rows = np.where(mask.max(axis=1))[0]
    cols = np.where(mask.max(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0: return {'prep': 'z-score', 'roi_coords': None, 'detections': []}
    
    y_start_img, y_end_img = rows[0], rows[-1] + 1
    x_start_img, x_end_img = cols[0], cols[-1] + 1
    rotated_img = rotated_img[y_start_img:y_end_img, x_start_img:x_end_img]

    x_center_rot = x_center_pix - x_start_img
    y_center_rot = y_center_pix - y_start_img

    # Eye orientation flip
    eye_coords = np.array([[[eye_x_avg, eye_y_avg]]], dtype=np.float32)
    eye_rotated = cv2.transform(eye_coords, M_rot)[0][0]
    eye_rotated[0] -= x_start_img
    eye_rotated[1] -= y_start_img

    did_flip = False
    h_c, w_c, _ = rotated_img.shape
    if eye_rotated[1] > y_center_rot:
        rotated_img = cv2.flip(rotated_img, 0)
        y_center_rot = h_c - y_center_rot
        did_flip = True

    # ROI Constraints
    half_width = int((width_frac * 1.5) * w / 2)
    roi_length = min(int(0.5 * h_c), h_c - int(y_center_rot)) 
    
    y_start = int(y_center_rot)
    y_end = int(min(h_c, y_start + roi_length))
    x_start = int(max(0, x_center_rot - half_width))
    x_end   = int(min(w_c, x_center_rot + half_width))

    roi = rotated_img[y_start:y_end, x_start:x_end]
    if roi.size == 0: return {'prep': 'z-score', 'roi_coords': None, 'detections': []}

    # U-Net Inference
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    input_img, ratio, pad_w, pad_h = letterbox(roi_gray, model_input_size)
    
    image_float = input_img.astype(np.float32)
    norm_image = (image_float - np.mean(image_float)) / (np.std(image_float) + 1e-8)
    
    tensor_img = torch.from_numpy(norm_image).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits).cpu().squeeze().numpy()

    shaft_prob, tip_prob = probs[0], probs[1]
    shaft_mask = cv2.dilate((shaft_prob > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8), iterations=1)
    masked_tip_prob = tip_prob * shaft_mask

    # Get max activation point
    py, px = np.unravel_index(np.argmax(masked_tip_prob), masked_tip_prob.shape)
    confidence = masked_tip_prob[py, px]

    # Inverse Geometry Translation
    roi_x = (px - pad_w) / ratio
    roi_y = (py - pad_h) / ratio
    rot_x = roi_x + x_start
    rot_y = roi_y + y_start
    if did_flip: rot_y = (h_c - 1) - rot_y
    rot_x += x_start_img
    rot_y += y_start_img

    pt = np.array([[[rot_x, rot_y]]], dtype=np.float32)
    M_inv = cv2.invertAffineTransform(M_rot)
    orig_pt = cv2.transform(pt, M_inv)[0][0]
    final_x, final_y = orig_pt

    # Construct synthetic YOLO Bounding Box (2% of image size)
    box_w = 0.02 * w
    box_h = 0.02 * h
    x_min = final_x - (box_w / 2)
    y_min = final_y - (box_h / 2)
    x_max = final_x + (box_w / 2)
    y_max = final_y + (box_h / 2)

    detections = [(7, x_min, y_min, x_max, y_max, confidence)] # Class 7 = Spina Tip
    roi_coords = (y_start, y_end, x_start, x_end)
    
    # --- Debug Plotting ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(6, 6))
    plt.imshow(masked_tip_prob, cmap='magma')
    plt.scatter([px], [py], c='cyan', marker='x', s=100)
    plt.axis('off')
    
    debug_path = os.path.join(os.path.dirname(image_path), f"debug_heatmap_{os.path.basename(image_path)}")
    plt.savefig(debug_path, bbox_inches='tight')
    plt.close()
    # ----------------------

    detections = [(7, x_min, y_min, x_max, y_max, confidence)] 
    roi_coords = (y_start, y_end, x_start, x_end)
    
    return {'prep': 'z-score', 'roi_coords': roi_coords, 'detections': detections}

# --- 4. File Management ---
def SaveExtraDetections(extra_detections, labels_file, roi_coords, img_shape):
    h, w = img_shape[:2]
    with open(labels_file, "a") as f:
        for det in extra_detections:
            cls_id, x1, y1, x2, y2, _ = det
            
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h

            f.write(f"{cls_id} {x_center} {y_center} {bbox_w} {bbox_h}\n")

def SpinaTipEnhance(image_path, label_file, model_path):
    if os.path.isdir(image_path) and os.path.isdir(label_file):
        valid_exts = (".jpg", ".jpeg", ".png")
        images = [f for f in os.listdir(image_path) if f.lower().endswith(valid_exts)]
        print(f"Found {len(images)} images in {image_path}")

        for img_name in images:
            img_path_full = os.path.join(image_path, img_name)
            lbl_path = os.path.join(label_file, os.path.splitext(img_name)[0] + ".txt")

            if not os.path.exists(lbl_path): continue
            SpinaTipEnhance(img_path_full, lbl_path, model_path)
        return 

    with open(label_file, "r") as f:
        labels_in_file = [line.strip().split() for line in f.readlines()]

    labels_in_file = [(int(float(l[0])), float(l[1]), float(l[2]), float(l[3]), float(l[4])) for l in labels_in_file]

    spina_base = None
    eyes = []
    spina_tip_detected = False
    
    for cls, x, y, w, h in labels_in_file:
        if cls == 6: spina_base = (x, y, w, h)
        elif cls == 3: eyes.append((x, y))
        elif cls == 7: spina_tip_detected = True

    if spina_base and len(eyes) > 0 and not spina_tip_detected:
        print(f"Enhancing: {label_file}")
        
        # model_path is passed instead of a loaded YOLO model
        results = RedetectSpinaTip(image_path, model_path, spina_base[:3], eyes)
        extra_detections = results['detections']
        roi_coords = results['roi_coords']

        if extra_detections:
            img_shape = cv2.imread(image_path).shape[:2]
            SaveExtraDetections(extra_detections, label_file, roi_coords, img_shape)
            print(" -> Tip added to file.")
        else:
            print(" -> No tip structure found.")




