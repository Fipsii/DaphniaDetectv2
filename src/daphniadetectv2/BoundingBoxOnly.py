# Import required modules from the CollectedCode package
from DetectorCode import NMS_Crop, NMS_detect, SaveData, ConvertToJPG
import os
import json
import pandas as pd
# ==========================
# CONFIGURATION & PARAMETERS
# ==========================

import os

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Output directory for storing results
OutputDir: str = script_dir + "/Detector"

# Paths to trained YOLO model weights
Bbox: str = os.path.join(script_dir, "Model/detect/weights/best.pt")  # Model for bounding box detection
Segment: str = os.path.join(script_dir, "Model/segment/weights/best.pt")  # Model for segmentation
Classify: str = os.path.join(script_dir, "Model/classify/weights/best.pt")  # Model for classification

# Directory containing input images
ImageDir: str = None
ImageDir = "/home/fipsi/Downloads/2025_Images_Sorted/Classification_flat"
# If no folder was selected request a path
if not ImageDir or not os.path.exists(ImageDir):
    ImageDir = input("Please enter the path to the image folder: ").strip()
    while not os.path.exists(ImageDir):
        print("Invalid path. Please try again.")
        ImageDir = input("Please enter the path to the image folder: ").strip()

if script_dir == os.path.dirname(os.path.abspath(__file__)):
    print(f"No save location specified saving results to {script_dir}")


# ======================================
# STEP 0.5: DETECT ORGANS IN THE IMAGES
# ======================================
## If Data is not JPG convert ##



#ConvertToJPG.ConvertToJPEG(ImageDir, ImageDir + "/JPG")
#ImageDir = ImageDir +"/JPG"



# ======================================
# STEP 1: DETECT ORGANS IN THE IMAGES
# ======================================
## DaphniaDetector missing scale

# Detect organs using the YOLO object detection model
# Parameters:
# - ImageDir (str): Directory containing images for processing
# - OutputDir (str): Directory where results will be saved
# - vis (bool): Whether to visualize detections
# - NMS (bool): Whether to apply Non-Maximum Suppression (NMS)
# - crop (bool): Whether to crop detected regions
# - ModelPath (str): Path to the trained YOLO model for detection


NMS_detect.DetectOrgans(ImageDir, OutputDir, vis=False, NMS=True, crop=True,organs = ["Daphnia"], ModelPath=Bbox)

# ======================================
# STEP 2: UPDATE BODY BOUNDING BOXES
# ======================================

# Update daphnid body bounding boxes to ensure they encompass all detected body parts
# Parameters:
# - label_dir (str): Directory containing YOLO detection labels
label_dir: str = os.path.join(OutputDir, "Detection", "labels")

for file in os.listdir(label_dir):
    label_path = os.path.join(label_dir, file)
    NMS_detect.update_daphnid_bounding_boxes(label_path)


# ======================================
# STEP 3: SAVE COORDINATES AND SAVE
# ======================================
# Save measurement results
# Parameters:
# - label_dir (folder): folder with labels
# - ImageDir (folder): folder with original images
# Returns:
# - (pd.dataframe): dataframe pixel values labels


BoundingBoxAnnotations = SaveData.read_yolo_folder(label_dir, ImageDir)
BoundingBoxAnnotationsPixel = SaveData.convert_yolo_to_pixel(BoundingBoxAnnotations)

# Merge the data bassed on the image name
BoundingBoxAnnotationsPixel.to_csv(f"{OutputDir}/BoundingBoxes.csv")






