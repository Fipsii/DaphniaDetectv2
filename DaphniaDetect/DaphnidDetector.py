# Import required modules from the CollectedCode package
from DetectorCode import NMS_Crop, NMS_detect, SegmentYOLODeploy, YOLODeploy, DataDict, ScaleDetect,LengthMeasure, ConvertToJPG
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


if ConvertToJPG.CheckJPG(ImageDir) == False:
    ConvertToJPG.ConvertToJPEG(ImageDir, ImageDir + "/JPG")
    ImageDir = ImageDir +"/JPG"



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


# _ Crop still has an error -> 
print(Bbox)
NMS_detect.DetectOrgans(ImageDir, OutputDir, vis=True, NMS=True, crop=False, ModelPath=Bbox)

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
# STEP 3: SEGMENTATION OF ORGANS
# ======================================

# Apply segmentation model to detected objects
# Parameters:
# - ImageDir (str): Directory containing input images
# - OutputDir (str): Directory to save segmentation results
# - Vis (bool): Whether to visualize segmentation results
# Returns:
# - (folder): folder in outputdir with segmentation results sepcified
SegmentYOLODeploy.Segment_Exp(ImageDir, OutputDir,ModelPath = Segment, Vis=True)

# ======================================
# STEP 4: CLASSIFY SPECIES
# ======================================

# Classify detected species using a trained classification model
# Parameters:
# - ImageDir (str): Directory containing input images
# - Classify (str): Path to the trained classification model
# Returns:
# - (dict): Dictionary mapping image names to classified species

species: dict = YOLODeploy.Classify_Species(ImageDir, Classify)

# ======================================
# STEP 5: MEASUREMENTS
# ======================================

# Measure width using Imhof method
# Parameters:
# - ImageDir (str): Directory containing input images
# - OutputDir (str): Directory to save measurement results
# Returns:
# - (dict): Dictionary containing measurement results
test: dict = DataDict.WidthImhof(ImageDir, OutputDir)
print(test)
# Alternative method: Measure width using Rabus method
# test = DataDict.WidthRabus(ImageDir, OutputDir)


# ======================================
# STEP 6: GET SCALE VALUES
# ======================================

# Visualize and save measurement results
# Parameters:
# - test (dict): Dictionary containing measurement results
# - Scale_detector_mode (int): Detection of no scale (0), uniform scale (1), heterogenous scale (2)
# Returns:
# - (pd.dataframe): dataframe with scale values (also saved under outputdir/scales.csv)

Scale_detector_mode = 2
Scales = ScaleDetect.DetectScale(test,Scale_detector_mode,Conv_factor=0)
print(Scales)
Scales.to_csv(OutputDir + "/scale.csv", index = False)      

## Add scale values to dict by image_name then we do not need to merge later

scales_dict = Scales.set_index("image_name").to_dict(orient="index")

combined_dict = {}
for key in set(scales_dict) | set(test):  # Union of keys
    combined_dict[key] = {**scales_dict.get(key, {}), **test.get(key, {})}
    
# ======================================
# STEP 7: VISUALIZE RESULTS
# ======================================

# Visualize and save measurement results
# Parameters:
# - test (dict): Dictionary containing measurement results
# - output_path (str): Path to save visualized images
# Returns:
# - (images): Images saved in OutputDir/visualization
DataDict.visualize_and_save(combined_dict, os.path.join(OutputDir, "visualization"), Scale_detector_mode)

# ======================================
# STEP 8: MEASURE LENGTH
# ======================================

# - test (dict): Dictionary containing measurement results
# Returns:
# - test (dict): Dictionary containing measurement results with length measurements
Measurements: dict = LengthMeasure.MeasureLength(combined_dict)


# ======================================
# STEP 9: CONVERT AND SAVE
# ======================================
# Visualize and save measurement results
# Parameters:
# - test (dict): Dictionary containing measurement results
# - Scale (dataframe): Dataframe with conversion factor
# Returns:
# - (pd.dataframe): dataframe with mm measurements (also saved under outputdir/scaled_measurements.csv)
Measurements = pd.DataFrame.from_dict(Measurements,orient='index')
print(Measurements)
# Merge the data bassed on the image name

merged_df = pd.merge(Measurements, Scales, on="image_name", how="inner") 
Measurements.to_csv(f"{OutputDir}/data.csv")

scaled_data = merged_df.apply(LengthMeasure.scale_values, axis=1).apply(pd.Series)
scaled_data.to_csv(f"{OutputDir}/scaled_measurements.csv", index=False)




