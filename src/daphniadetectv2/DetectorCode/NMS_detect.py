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

def DetectOrgans(Images,OutputDir, vis=True, NMS=True, crop=False, organs = ["Eye"],  ModelPath="StandardPath", conf = 0.01, iou = 0.2):
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

