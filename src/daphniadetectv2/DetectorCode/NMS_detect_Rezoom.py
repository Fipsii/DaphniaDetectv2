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

def DetectOrgans(Images,OutputDir, vis=True, NMS=True, crop=False, refineTip=True, organs = ["Eye"],  ModelPath= None, SpinaModelPath = None,conf = 0.01, iou = 0.2):
    '''
    Detect specified organs in a list of images using a YOLO model and save the detection results.

    Parameters:
        Images (list): A list of image file paths to be processed.
        OutputDir (str): Directory to save the output files (e.g., annotated images, labels).
        vis (bool, optional): If True, saves annotated images with bounding boxes. Defaults to True.
        NMS (bool, optional): If True, applies Non-Maximum Suppression (NMS) to filter overlapping detections. Defaults to True.
        crop (bool, optional): If True, crops detected organs and saves them separately. Defaults to False.
        organs (list, optional): List of organ class names to filter and detect (e.g., ["Eye"]). Defaults to ["Eye"].
        ModelPath (str, optional): Path to the trained YOLO model to use for inference. Defaults to "StandardPath".
        conf (float, optional): Confidence threshold for detections. Defaults to 0.01.
        iou (float, optional): IoU (Intersection over Union) threshold for NMS. Defaults to 1 (no suppression).
        
        Important: We manually employ additional NMS later in which we filter for one instance per object

    Returns:
        Body (dict): Dictionary mapping image filenames to detection metadata.
        Also saves:
            - YOLO-formatted .txt files with annotations.
            - (Optionally) annotated images and/or cropped organ images in OutputDir.
    '''

    obj_detect_start = time.time()

    # Initialize YOLO model
    model = YOLO(ModelPath)

    # Run YOLO model
    results = model(Images, stream=True, imgsz = 1280,  conf=conf, iou = iou, project=OutputDir, name="Detection", verbose=False)
    obj_detect_end = time.time()
    
    # Create storage folder:
    os.makedirs(OutputDir + "/Detection/labels", exist_ok=True)

    for result in results:
      boxes = result.boxes # Boxes object for bounding box outputs
      classes = boxes.cls
      conf = boxes.conf
      boxes = boxes.xywhn
      
      if NMS == True:
        
        # Reshape the classes and conf tensors to match the box tensor shape for concatenation
        classes = classes.unsqueeze(1)  # Shape becomes (2, 1)
        conf = conf.unsqueeze(1)  # Shape becomes (2, 1)
        
        # Concatenate along the last axis (columns)
        combined = torch.cat((classes, boxes, conf), dim=1)  # Shape: (2, 6)
        
        # List to store final rows with highest confidence for each class
        final_rows = []
        
        # Keep track of the classes we've already seen
        seen_classes = set()
        
        # Iterate through the rows
        for row in combined:
            class_id = row[0].item()  # First column is the class
            if class_id not in seen_classes:
                final_rows.append(row)  # Keep the first occurrence (highest confidence)
                seen_classes.add(class_id)  # Mark the class as seen
        
        # Stack the final rows together
        final_tensor = torch.stack(final_rows)
        
        # Now we get rid of last column and write into a textfile
        final_tensor = final_tensor[:, :-1]
           
      else:
        classes = classes.unsqueeze(1)  # Shape becomes (2, 1)
        # Concatenate along the last axis (columns)
        
        final_tensor = torch.cat((classes, boxes), dim=1)  # Shape: (2, 6)
      label_saveloc = f"{OutputDir}/Detection/labels/{os.path.splitext(os.path.basename(result.path))[0]}.txt"
      # Open a file to write the tensor
      with open(label_saveloc, "w") as f:
        # Iterate through each row of the tensor and write it as a line
        for row in final_tensor:
            # Convert each row to a space-separated string and write to file
            f.write(" ".join(map(str, row.tolist())) + "\n")
    
    labels_folder = os.path.join(OutputDir, "Detection", "labels")
    for filename in os.listdir(labels_folder):
        file_path = os.path.join(labels_folder, filename)
        update_daphnid_bounding_boxes(file_path)
  
    if refineTip:
        print("\nDetection of missing spina tips with specialised model")

     
        SpinaTipEnhance(Images, labels_folder, SpinaModelPath)

      
         ## This is were the function needs to fit 
   
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
        # Draw boxes from saved YOLO labels
        DrawYOLOBoxes(
            Original_Images=Images,
            labels_folder=labels_folder,
            Save_folder=os.path.join(OutputDir, "Detection", "visuals"),
            class_mapping=result.names
        )
    
    
    print(f"\nAnnotations saved to: {OutputDir}")
    print("Please review the results for any potential issues.\n")

    return OutputDir  # Return path where annotations are saved

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


def RedetectSpinaTip(image_path, model, spina_base, eyes, model_input_size=640):
    """
    Generate an ROI along the eye-spina base axis (non-axis-aligned), 
    apply CLAHE, run YOLO, return only the highest-confidence detection,
    and show the rotated ROI crop with axis and mapped detection box.
    """
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    # --- Load image ---
    print(image_path)
    img = cv2.imread(image_path)
    h, w, _ = img.shape

    # --- Spina base in pixels ---
    x_center, y_center, width_frac = spina_base
    x_center_pix, y_center_pix = int(x_center * w), int(y_center * h)

    # --- Average eye position in pixels ---
    eye_x_avg = np.mean([x for x, y in eyes]) * w
    eye_y_avg = np.mean([y for x, y in eyes]) * h

    # --- Rotation matrix (align eye-base axis vertical) ---
    dx, dy = eye_x_avg - x_center_pix, eye_y_avg - y_center_pix
    angle = np.degrees(np.arctan2(dy, dx))
    M_rot = cv2.getRotationMatrix2D((x_center_pix, y_center_pix), angle + 90, 1.0)
    rotated_img = cv2.warpAffine(img, M_rot, (w, h))

    # --- Crop out fully black rows/cols from rotated image ---
    mask = rotated_img.sum(axis=2) > 0
    rows = np.where(mask.max(axis=1))[0]
    cols = np.where(mask.max(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        raise ValueError("Rotated image is completely black!")

    y_start_img, y_end_img = rows[0], rows[-1] + 1
    x_start_img, x_end_img = cols[0], cols[-1] + 1
    rotated_img = rotated_img[y_start_img:y_end_img, x_start_img:x_end_img]

    # --- Update spina base and eye coordinates after crop ---
    x_center_rot = x_center_pix - x_start_img
    y_center_rot = y_center_pix - y_start_img

    eye_coords = np.array([[[eye_x_avg, eye_y_avg]]], dtype=np.float32)
    eye_rotated = cv2.transform(eye_coords, M_rot)[0][0]
    eye_rotated[0] -= x_start_img
    eye_rotated[1] -= y_start_img

    # --- Flip if eye is below base (always keep eye "above") ---
    did_flip = False
    h_c, w_c, _ = rotated_img.shape
    if eye_rotated[1] > y_center_rot:
        rotated_img = cv2.flip(rotated_img, 0)
        y_center_rot = h_c - y_center_rot
        eye_rotated[1] = h_c - eye_rotated[1]
        did_flip = True

    # --- Define ROI ---
    half_width = int((width_frac*1.5) * w / 2)
    roi_length = min(int(0.5 * h_c), h_c - int(y_center_rot)) 
    
    y_start = y_center_rot
    y_end = min(h_c, y_start + roi_length)
    
    x_start = max(0, x_center_rot - half_width)
    x_end   = min(w_c, x_center_rot + half_width)

 
    roi = rotated_img[y_start:y_end, x_start:x_end]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        raise ValueError("ROI out of bounds or zero-sized")

    # Resize to model input size
    roi_resized, scale, pad_x, pad_y = resize_with_zoom(roi, model_input_size)

    roi_smooth = cv2.GaussianBlur(roi_resized, (7, 7), sigmaX=0.5, sigmaY=0.5)

    # Apply CLAHE
    lab = cv2.cvtColor(roi_smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2, 2))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    roi_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    plt.figure(figsize=(8,8))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(roi_clahe)
    
    # --- Show original image with axis and ROI ---
    plt.figure(figsize=(8,8))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.plot([x_center_pix, eye_x_avg], [y_center_pix, eye_y_avg], 'b-', linewidth=2, label='Eye-Spina axis')

    # ROI polygon in original image
    corners = np.array([[x_start, y_start], [x_end, y_start],
                        [x_end, y_end], [x_start, y_end]], dtype=np.float32)
    if did_flip:
        corners[:, 1] = (h_c - 1) - corners[:, 1]
    corners_uncropped = corners + np.array([x_start_img, y_start_img], dtype=np.float32)
    M_inv = cv2.invertAffineTransform(M_rot)
    roi_corners_orig = cv2.transform(corners_uncropped[None, :, :], M_inv)[0]
    plt.plot(np.r_[roi_corners_orig[:, 0], roi_corners_orig[0, 0]],
             np.r_[roi_corners_orig[:, 1], roi_corners_orig[0, 1]],
             'r-', linewidth=2, label='ROI')
             
    plt.legend(); plt.axis('off'); plt.show()
    
    roi_h, roi_w = roi_clahe.shape[:2]
    width = roi_w               # size of square
    half_width = width // 2  # stride

    roi_clahe_segments = []
    crop_coords = []
    
    roi_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    roi_rgb = cv2.cvtColor(roi_clahe, cv2.COLOR_BGR2RGB)
    
    for y in range(0, roi_h - width + 1, half_width):
     for x in range(0, roi_w - width + 1, half_width):
        crop = roi_rgb[y:y+width, x:x+width]

        # Skip empty or too-small crops (safety check)
        if crop.size == 0 or crop.shape[0] != width or crop.shape[1] != width:
            continue

        roi_clahe_segments.append(crop)
        crop_coords.append((x, y, x+width, y+width))
    
    import matplotlib.pyplot as plt

    # Number of crops
    n_segments = len(roi_clahe_segments)

    # Decide grid size (square-like layout)
    cols = int(np.ceil(np.sqrt(n_segments)))
    rows = int(np.ceil(n_segments / cols))

    plt.figure(figsize=(20, 20))

    for i, seg in enumerate(roi_clahe_segments):
     plt.subplot(rows, cols, i+1)
     plt.imshow(seg)
     plt.title(f"Seg {i}")
     plt.axis("off")

    plt.tight_layout()
    plt.show()

    # --- Run YOLO ---
    model = YOLO(model)  # load your model
    all_results = []

    for i, seg in enumerate(roi_clahe_segments):
     results = model(seg, imgsz=model_input_size, conf=0.1, iou=0.2, save=True, verbose=False)
     if len(results[0].boxes) > 0:  # if any detections
        confs = results[0].boxes.conf.cpu().numpy()
        best_conf = float(np.max(confs))
        all_results.append((i, crop_coords[i], best_conf, results))

    # --- Pick the crop with highest confidence ---
    if all_results:
     best_idx, best_coords, best_conf, best_results = max(all_results, key=lambda x: x[2])
     print(f"Best crop: Seg {best_idx}, Conf={best_conf:.3f}, Coords={best_coords}")
     best_results[0].show()  # show YOLO detection
    else:
     print("No detections found in any crop.")

    # --- After YOLO ---
    best_det, best_conf = None, -1
    for result in results:
        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            if conf.item() > best_conf:
                best_conf = conf.item()
                best_det = (cls.item(), box, conf.item())

    detections = []
    if best_det is not None:
        cls, box, conf = best_det
        x1, y1, x2, y2 = [float(v.cpu().item()) if hasattr(v, 'cpu') else float(v) for v in box]

        # Undo resize_with_zoom scaling & padding
        x1 = (x1 - pad_x) / scale
        x2 = (x2 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        y2 = (y2 - pad_y) / scale

        # 4-corner box in ROI coords
        box_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

        # Shift into rotated image coords (account for ROI origin)
        box_pts[:, 0] += x_start
        box_pts[:, 1] += y_start

        # Undo flip if applied
        if did_flip:
            box_pts[:, 1] = (h_c - 1) - box_pts[:, 1]

        # Add crop offset (uncropped rotated space)
        box_pts_uncropped = box_pts + np.array([x_start_img, y_start_img], dtype=np.float32)

        # Back-transform with inverse rotation to original image coords
        box_pts_orig = cv2.transform(box_pts_uncropped[None, :, :], M_inv)[0]
        x_min, y_min = box_pts_orig.min(axis=0)
        x_max, y_max = box_pts_orig.max(axis=0)
        detections.append((cls, x_min, y_min, x_max, y_max, conf))

    # --- Draw final detections ---
    plt.figure(figsize=(8,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Draw eye & spina base
    plt.scatter([x_center_pix], [y_center_pix], c='blue', label='Spina base')
    if eyes:
        eye_xs, eye_ys = zip(*eyes)
        plt.scatter([x for x, y in eyes], [y for x, y in eyes], c='yellow', label='Eyes')
        plt.scatter([x * w for x in eye_xs], [y * h for y in eye_ys], c='green', label='Eyes')

    # Draw detection box

    for cls, x_min, y_min, x_max, y_max, conf in detections:
     # Reconstruct polygon for this detection
     box_pts_orig = np.array([[x_min, y_min],
                             [x_max, y_min],
                             [x_max, y_max],
                             [x_min, y_max]], dtype=np.float32)
     # Close the rectangle
     box_closed = np.vstack([box_pts_orig, box_pts_orig[0]])
     plt.plot(box_closed[:, 0], box_closed[:, 1], 'r-', linewidth=2, label='Detected Spina Tip')


    plt.legend(); plt.axis('off'); plt.show()

    roi_coords = (y_start, h, x_center_pix - half_width, x_center_pix + half_width)
    return {'prep': 'clahe', 'roi_coords': roi_coords, 'detections': detections}





def SaveExtraDetections(extra_detections, labels_file, roi_coords, img_shape):
    """
    Saves zoomed detections in YOLO format to append to original labels.
    extra_detections: list of (cls_id, x1, y1, x2, y2) in full image coordinates
    roi_coords: (y_start, y_end, x_start, x_end) of cropped ROI
    img_shape: (height, width) of original image
    """
    h, w = img_shape[:2]
    y_start, y_end, x_start, x_end = roi_coords
   
    with open(labels_file, "a") as f:
        for det in extra_detections:
            _, x1, y1, x2, y2, _ = det
            # Convert tensors to float if needed
            x1, y1, x2, y2 = [v.item() if isinstance(v, torch.Tensor) else v for v in (x1, y1, x2, y2)]
            
            # Normalize relative to original image
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h

            f.write(f"7 {x_center} {y_center} {bbox_w} {bbox_h}\n")




import os
import cv2

def SpinaTipEnhance(image_path, label_file, model):
    """
    Detects Spina tip in images that have Spina base and Eyes but no Spina tip.
    Updates the YOLO label file with extra detections.

    Parameters:
        image_path (str): Path to the image OR folder of images.
        label_file (str): Path to the YOLO label file OR folder of label files.
        model: Model used for re-detecting the Spina tip.

    Returns:
        None. Updates the label file(s) in-place.
    """

    # Case 1: If given a folder, loop through all images + labels
    if os.path.isdir(image_path) and os.path.isdir(label_file):
        valid_exts = (".jpg", ".jpeg", ".png")
        images = [f for f in os.listdir(image_path) if f.lower().endswith(valid_exts)]
        print(f"Found {len(images)} images in {image_path}")

        for img_name in images:
            img_path = os.path.join(image_path, img_name)
            lbl_path = os.path.join(label_file, os.path.splitext(img_name)[0] + ".txt")

            if not os.path.exists(lbl_path):
                print(f" Skipping {img_name}, no label file found.")
                continue

            SpinaTipEnhance(img_path, lbl_path, model)  # recursive call

        return  # stop after processing folder

    # Case 2: Single image + label
    with open(label_file, "r") as f:
        labels_in_file = [line.strip().split() for line in f.readlines()]

    # Convert strings to floats/ints
    labels_in_file = [(int(l[0]), float(l[1]), float(l[2]), float(l[3]), float(l[4])) 
                      for l in labels_in_file]

    # Extract spina base, eyes, and check spina tip
    spina_base = None
    eyes = []
    spina_tip_detected = False
    for cls, x, y, w, h in labels_in_file:
        if cls == 6:  # Spina base
            spina_base = (x, y, w, h)
        elif cls == 3:  # Eye
            eyes.append((x, y))
        elif cls == 7:  # Spina tip
            spina_tip_detected = True

    # Only run enhancement if base + eyes exist, but tip is missing
    if spina_base and len(eyes) > 0 and not spina_tip_detected:
        print(f"Running SpinaTipEnhance for {label_file}...")
        
        small_detect = RedetectSpinaTip(image_path, model, spina_base[:3], eyes)
        extra_detections = small_detect['detections']
        roi_coords = small_detect['roi_coords']

        # Get image shape
        img_shape = cv2.imread(image_path).shape[:2]

        # Save new detections into label file
        SaveExtraDetections(extra_detections, label_file, roi_coords, img_shape)







