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
                if base_name in os.path.basename(ann):
                    ann_path = ann
                    break

            if ann_path is None:
                print(f"Annotation file for {img_name} not found.")
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

def DetectOrgans(Images,OutputDir, vis=True, NMS=True, crop=False, organs = ["Eye"],  ModelPath="StandardPath"):
    ''' 
    Detect the organs and save the annotations in YOLO format.
    
    Input: list(Images), str(ModelPath), str(OutputDir)
    Output: YOLO formatted .txt annotation files, Body
    '''

    obj_detect_start = time.time()

    # Initialize YOLO model
    model = YOLO(ModelPath)

    # Run YOLO model
    results = model(Images, stream=True, imgsz = 1280, save = vis,  conf=0.25,project=OutputDir, name="Detection", verbose=False)
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
      
    print("Now Cropping images") 
    if crop == True:
      
      CropImagesFromYOLO(Images,labels_folder=OutputDir + "/Detection/labels",Crop_mode = organs, Save_folder=OutputDir+"/Detection/crops",class_mapping=result.names)
        
    print(f"Annotations stored in {OutputDir}. We advise checking for errors.")
    return OutputDir  # Return path where annotations are saved

def update_daphnid_bounding_boxes(annotation_file):
    """
    Reads a YOLO annotation file and updates all Daphnid body bounding boxes (class_id == 6)
    to be enclosed by the largest detected Daphnid bounding box.

    :param annotation_file: Path to the YOLO annotation file.
    """
    if not os.path.exists(annotation_file):
        print(f"Annotation file {annotation_file} does not exist.")
        return

    # Read the annotations
    with open(annotation_file, 'r') as f:
        lines = f.readlines()

    # Initialize max values
    max_x_center, max_y_center, max_width, max_height = None, None, None, None

    # Step 1: Find the largest enclosing Daphnid body box
    for line in lines:
        parts = line.strip().split()

        try:
            class_id = int(float(parts[0]))  # Fix: Convert potential '4.0' to 4
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            print(f"Skipping malformed line in {annotation_file}: {line}")
            continue

        if class_id == 6:  # Daphnid body class
            if max_x_center is None:
                max_x_center, max_y_center, max_width, max_height = x_center, y_center, width, height
            else:
                max_x_center = min(max_x_center, x_center)  # Smallest center x
                max_y_center = min(max_y_center, y_center)  # Smallest center y
                max_width = max(max_width, width)  # Largest width
                max_height = max(max_height, height)  # Largest height

    # If no Daphnid body was found, exit
    if max_x_center is None:
        print(f"No Daphnid body found in {annotation_file}")
        return

    # Step 2: Update all Daphnid bounding boxes to the largest one
    updated_lines = []
    for line in lines:
        parts = line.strip().split()

        try:
            class_id = int(float(parts[0]))  # Fix: Convert potential '4.0' to 4
        except ValueError:
            updated_lines.append(line)
            continue

        if class_id == 6:
            updated_lines.append(f"{class_id} {max_x_center:.6f} {max_y_center:.6f} {max_width:.6f} {max_height:.6f}\n")
        else:
            updated_lines.append(line)

    # Step 3: Write back the updated annotations
    with open(annotation_file, 'w') as f:
        f.writelines(updated_lines)

