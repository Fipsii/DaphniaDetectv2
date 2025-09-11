# Import required modules from the CollectedCode package
from DetectorCode import NMS_Crop, NMS_detect_Rezoom, SegmentYOLODeploy, YOLODeploy, DataDict, ScaleDetect,LengthMeasure, ConvertToJPG
import os
import json
import pandas as pd
# ==========================
# CONFIGURATION & PARAMETERS
# ==========================
import time
import os

start = time.time()

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Output directory for storing results
OutputDir: str = script_dir + "/Detector"

# Paths to trained YOLO model weights
Bbox: str = os.path.join(script_dir, "Model/detect/weights/best.pt")  # Model for bounding box detection
Segment: str = os.path.join(script_dir, "Model/segment/weights/best.pt")  # Model for segmentation
Classify: str = os.path.join(script_dir, "Model/classify/weights/best.pt")  # Model for classification
SpinaModel: str = "/home/fipsi/Downloads/AllCol28/weights/best.pt"  # Model for classification

# Directory containing input images
ImageDir: str = None

# Should be classified?
Classify_Species = True

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


ConvertToJPG.ConvertToJPEG(ImageDir, ImageDir + "/JPG")
ImageDir = ImageDir +"/JPG"



# ======================================
# STEP 1+2: DETECT ORGANS IN THE IMAGES
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

NMS_detect_Rezoom.DetectOrgans(ImageDir, OutputDir, vis=True, NMS=True, crop=True,refineTip = False,organs = ["Heart","Daphnia", "Eye", "Spina tip", "Spina base"], ModelPath=Bbox, SpinaModelPath=SpinaModel)


# ======================================
# STEP 3: SEGMENTATION
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

if Classify_Species == True:
 DaphniaCropDir = OutputDir + "/Detection/crops/Daphnia"
 species: dict = YOLODeploy.Classify_Species(DaphniaCropDir, Classify)

# ======================================
# STEP 5: MEASUREMENTS
# ======================================

# Measure width using Imhof method
# Parameters:
# - ImageDir (str): Directory containing input images
# - OutputDir (str): Directory to save measurement results
# - Method (str): Sperfeld, Rabus or Imhof
# 	Imhof: broadest point straight Daphnia (not perpendicular to eye-spina axis)
# 	Sperfeld: Perpendicular distance between ventral and dorsal midpoints of the body
# 	Rabus: maximum distance between the dorsal and the ventral edge of the carapace
# Returns:
# - (dict): Dictionary containing measurement results


test: dict = DataDict.BodyWidthMeasure(ImageDir, OutputDir, Method="Sperfeld")


# ======================================
# STEP 6: GET SCALE VALUES
# ======================================

# Visualize and save measurement results
# Parameters:
# - test (dict): Dictionary containing measurement results
# - Scale_detector_mode (int): Detection of no scale (0), uniform scale (1), heterogenous scale (2)
# Returns:
# - (pd.dataframe): dataframe with scale values (also saved under outputdir/scales.csv)

Scale_detector_mode = 1
Scales = ScaleDetect.Scale_Measure(test,Scale_detector_mode,Conv_factor=2.139)

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

# Merge the data bassed on the image name

## Problem data.csv has image_name empty for some reason
##

# Rename the column with .jpg extensions to image_name while image_name is renamed to 
# image_name_no_ext(ension)
Measurements = Measurements.reset_index()
Measurements = Measurements.rename(columns={'image_name': 'image_name_no_ext'})
Measurements.columns.values[0] = 'image_name'

if species:
	species_df = pd.DataFrame(list(species.items()), columns=['image_name', 'species'])
	Measurements = Measurements.merge(species_df, on='image_name', how='left')

Measurements.to_csv(f"{OutputDir}/data.csv")


scaled_data = Measurements.apply(LengthMeasure.scale_values, axis=1).apply(pd.Series)
scaled_data.to_csv(f"{OutputDir}/scaled_measurements.csv", index=False)


end = time.time()
print(f"Elapsed time: {end - start:.4f} seconds")

