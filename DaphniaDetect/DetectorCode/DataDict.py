import os
import numpy as np
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import imutils
import json
from pathlib import Path
import cv2 as cv
from math import atan2, cos, sin, sqrt, pi
import numpy as np
import math

def load_yolo_annotations(annotation_file):
    """
    Load YOLO annotations from a text file.
    
    :param annotation_file: Path to the YOLO annotation text file.
    :return: List of annotations (either bounding boxes or masks).
    """
    if not os.path.exists(annotation_file):
        return []
    
    annotations = []
    with open(annotation_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            annotations.append((class_id, x_center, y_center, width, height))
    return annotations

def process_image_folder(InputDir, OutputDir):
    """
    Processes all images in InputDir, loads YOLO segmentation & detection annotations,
    converts them to pixel coordinates, and returns them in a dictionary.

    :param InputDir: Path to the input images directory.
    :param OutputDir: Path where YOLO annotations are stored.
    :return: Dictionary with image filenames as keys and their pixel-based annotations.
    """
    image_dir = InputDir
    segmentation_dir = os.path.join(OutputDir, "Segmentation", "labels")
    detection_dir = os.path.join(OutputDir, "Detection", "labels")

    results = {}

    for image_file in os.listdir(image_dir):
        if image_file.endswith((".jpg", ".png")):
            image_name = os.path.splitext(image_file)[0]
            image_path = os.path.join(image_dir, image_file)

            # Load image to get dimensions
            image = cv2.imread(image_path)

            if image is None:
                continue
            image_shape = image.shape  # (height, width, channels)

            # Load YOLO annotations
            segmentation_path = os.path.join(segmentation_dir, f"{image_name}.txt")
            detection_path = os.path.join(detection_dir, f"{image_name}.txt")

            # Handle missing or empty annotations
            segmentation_masks = load_yolo_annotations(segmentation_path) or []
            bounding_boxes = load_yolo_annotations(detection_path) or []

            # **Remove the first value from segmentation if present**
            if segmentation_masks and isinstance(segmentation_masks[0], list) and segmentation_masks[0][0] == 0:
                segmentation_masks = [seg[1:] for seg in segmentation_masks]  # Remove first value from each mask

            # Only process if there are bounding boxes
            if bounding_boxes:
                largest_mask, bbox_pixel = convert_annotations_to_pixel(image_shape, segmentation_masks, bounding_boxes)
            else:
                largest_mask, bbox_pixel = None, []  # No annotations for this image

            # Create an empty black image
            black_image = np.zeros_like(image, dtype=np.uint8)  # Create a black image (single channel)
            
            if largest_mask is not None:
                # Convert mask coordinates from (y, x) to (x, y) if necessary
                #largest_mask = largest_mask[:, [1, 0]]  # Swap columns to correct the coordinates Is this correct?
            
                # Ensure the largest mask is in the right format (int32, and reshape for fillPoly)
                largest_mask = largest_mask.astype(np.int32)
                largest_mask = largest_mask.reshape((-1, 1, 2))
            
                # Fill the polygon area with white (255) on the black image
                cv2.fillPoly(black_image, [largest_mask], (255))  # 255 for white color (single channel)
                        # Initialize the dictionary for this image
                        
            results[image_file] = {
                "image_name": image_name,
                "image_path": image_path,
                "mask": black_image,
                "bboxes": {}
            }

           

            # Process bounding boxes
            for class_id, x_min, y_min, x_max, y_max in bbox_pixel:
                # Draw the bounding boxes on the black image
                # Define the organ name based on the class_id
                class_names = {
                    0: "Head",
                    1: "Eye",
                    2: "Spina base",
                    3: "Spina tip",
                    4: "Body",
                    5: "Heart",
                    6: "Daphnia",
                    7: "Brood cavity"
                }
                class_name = class_names.get(class_id, "Unknown")

                # Add bounding box to the image's results
                if class_name not in results[image_file]["bboxes"]:
                    results[image_file]["bboxes"][class_name] = []

                results[image_file]["bboxes"][class_name].append({
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                    "x_center": (x_min + x_max) / 2,
                    "y_center": (y_min + y_max) / 2
                })
            
            
            # Save the resulting image with bounding boxes and masks
            output_image_path = os.path.join(OutputDir, "Processed_Images", f"{image_name}_processed.png")
            os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
            cv2.imwrite(output_image_path, black_image)

    return results

def load_yolo_annotations(file_path):
    """
    Loads YOLO annotations from a given text file. It can load both bounding box and 
    segmentation annotations depending on the file contents.

    :param file_path: Path to the YOLO annotation file (.txt)
    :return: List of parsed annotations (bounding boxes or segmentation masks)
    """
    annotations = []

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found - {file_path}")
        return annotations
    
    # Open the annotation file
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing spaces
            if line:
                parts = line.split()
                if len(parts) == 5:
                    # Format: <class_id> <x_center> <y_center> <width> <height> (for bounding boxes)
                    class_id, x_center, y_center, width, height = map(float, parts)
                    annotations.append({
                        "class_id": int(class_id),
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height
                    })
                else:
                    # Format: <class_id> <polygon_coordinates> (for segmentation)
                    class_id = int(parts[0])
                    polygon = list(map(float, parts[1:]))
                    annotations.append({
                        "class_id": class_id,
                        "polygon": polygon
                    })

    return annotations

def convert_annotations_to_pixel(image_shape, mask_rel, bbox_rel):
    """
    Converts segmentation masks and bounding boxes from relative to pixel coordinates.

    :param image_shape: Tuple (height, width) of the image.
    :param mask_rel: List of segmentation masks in relative coordinates.
    :param bbox_rel: List of bounding boxes in relative format [(class_id, x_center, y_center, w, h)].
    :return: Tuple (mask_pixel, bbox_pixel) with converted coordinates.
    """
    img_h, img_w = image_shape[:2]
    # Convert segmentation masks to pixel coordinates
    mask_pixel = []
    for mask in mask_rel:
        # Mask is a polygon, so the coordinates are given in pairs of (x, y)
        pixel_mask = np.array([[int(y * img_h), int(x * img_w)] for x, y in zip(mask["polygon"][::2], mask["polygon"][1::2])], dtype=np.int32)

        
        mask_pixel.append(pixel_mask)

    # Select the largest mask (if any)
    largest_mask = max(mask_pixel, key=lambda p: cv2.contourArea(p), default=None)

    # Convert bounding boxes to pixel coordinates (with class label)
    bbox_pixel = []
    for bbox in bbox_rel:
        class_id = bbox['class_id']
        x_center = bbox['x_center']
        y_center = bbox['y_center']
        w = bbox['width']
        h = bbox['height']
        # Convert from relative to pixel coordinates
        x_min = int((x_center - w / 2) * img_w)
        y_min = int((y_center - h / 2) * img_h)
        x_max = int((x_center + w / 2) * img_w)
        y_max = int((y_center + h / 2) * img_h)

        # Append the bounding box with class_id
        bbox_pixel.append((class_id, x_min, y_min, x_max, y_max))

    return largest_mask, bbox_pixel

# Mask alread exist in OutputDir Segmentation/Mask add them to the dict resulting process image folder as mask:

# Example usage


    
#print(test)
''' This process needs then exit a list sor lete draw_boxes run over the whole dict and save a list'''

## Need position and name of Eye, Spind base and Spina tip best case make df out of it
## We get a df image_name class id (We can transform that with results.names which we know for our model always)

# Class ID to Name dictionary
## Functions that need adaptation:
# Image_Rotation, PerpendicularLine_Eye_Sb


##### Body width after Imhoff 2017      #####
#############################################

### Based on segment output of daphnid_instances_0.1
### Measure a defined point: Halfway between eye and spina base and not only the longest
### Read in output from Object detection


def getOrientation(pts, img, visualize=False):
  
  
  ## [pca]
  # Construct a buffer used by the pca analysis
  sz = len(pts)
  data_pts = np.empty((sz, 2), dtype=np.float64)
  for i in range(data_pts.shape[0]):
    data_pts[i,0] = pts[i,0,0]
    data_pts[i,1] = pts[i,0,1]
 
  # Perform PCA analysis
  mean = np.empty((0))
  mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
  
  # Store the center of the object
  cntr = (int(mean[0,0]), int(mean[0,1]))
  ## [pca]
 
  angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
  
  if visualize == True:
    ###### Show what happens
    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (255, 255, 0), 1)
    drawAxis(img, cntr, p2, (0, 0, 255), 5)
   
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    ## [visualization]
   
    # Label with the rotation angle
    label = "  Rotation Angle: " + str(-int(np.rad2deg(angle)) - 90) + " degrees"
    textbox = cv.rectangle(img, (cntr[0], cntr[1]-25), (cntr[0] + 250, cntr[1] + 10), (255,255,255), -1)
    cv.putText(img, label, (cntr[0], cntr[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv.LINE_AA)
  return angle

def Image_Rotation(test_dict, visualize=False):
    """
    Rotates image masks based on contours found within the mask. 
    Stores the rotation angle and rotated mask back into the dictionary.

    :param test_dict: Dictionary containing image names and their corresponding mask data.
    :param visualize: Boolean flag to visualize rotated masks.
    :return: None (modifies test_dict in place)
    """
    
    for image_name, data in test_dict.items():
        
        mask = data.get("mask")  # Get the mask for the image
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask is None or np.sum(mask) == 0:  # Check if mask is empty
            continue  

        # Find all contours in the mask
        contours, _ = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

        # Initialize variables
        angle_deg = 0
        rotated_mask = mask  # Default to original if no valid contours found

        for contour in contours:
            if cv.contourArea(contour) < 3700:  # Ignore small contours
                continue

            # Calculate the orientation of the contour
            temp_angle = getOrientation(contour, mask)  # Get orientation of the contour
            angle_deg = -int(np.rad2deg(temp_angle)) - 90  # Convert to degrees

            # Rotate the mask based on the calculated angle
            rotated_mask = imutils.rotate_bound(mask, angle_deg)
            break  # Only use the first valid contour

        # Store results in the dictionary
        data["angle"] = angle_deg
        data["rotated_mask"] = rotated_mask


        # Optional visualization
        if visualize:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.title(f"Original Mask - {image_name}")
            plt.imshow(mask, cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title(f"Rotated Mask ({angle_deg}°) - {image_name}")
            plt.imshow(rotated_mask, cmap="gray")

            plt.show()

def drawAxis(img, p_, q_, color, scale):
  
  from math import atan2, cos, sin, sqrt, pi
  import cv2 as cv
  p = list(p_)
  q = list(q_)
 
  ## [visualization1]
  angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
  hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
  # Here we lengthen the arrow by a factor of scale
  q[0] = p[0] - scale * hypotenuse * cos(angle)
  q[1] = p[1] - scale * hypotenuse * sin(angle)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  # create the arrow hooks
  p[0] = q[0] + 9 * cos(angle + pi / 4)
  p[1] = q[1] + 9 * sin(angle + pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
  p[0] = q[0] + 9 * cos(angle - pi / 4)
  p[1] = q[1] + 9 * sin(angle - pi / 4)
  cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
  ## [visualization1]

def point_trans(ori_point, angle, ori_shape, new_shape):
    
    # Transfrom the point from original to rotated image.
    # Args:
    #    ori_point: Point coordinates in original image.
    #    angle: Rotate angle in radians.
    #    ori_shape: The shape of original image.
    #    new_shape: The shape of rotated image.
    # Returns:
    #    Numpy array of new point coordinates in rotated image.
    


    dx = ori_point[0] - ori_shape[1] / 2.0
    dy = ori_point[1] - ori_shape[0] / 2.0

    t_x = round(dx * math.cos(angle) - dy * math.sin(angle) + new_shape[1] / 2.0)
    t_y = round(dx * math.sin(angle) + dy * math.cos(angle) + new_shape[0] / 2.0)
    return np.array((int(t_x), int(t_y)))


def Detect_Midpoint(data_dict, debug = False, outputdir = False):
    """
    Detects the midpoint between the eye and spina base in rotated images.
    Visualizes the eye, spina base, and midpoint on the rotated image.

    :param data_dict: Dictionary containing image metadata, including rotated masks, bounding boxes, angles, and original masks.
    :param output_folder: Directory to save visualized images.
    :return: Updated data_dict with detected midpoints stored under ["Mid_Points"].
    """

    for image_key, item in data_dict.items():
        img = item.get("rotated_mask")
        bboxes = item.get("bboxes", {})  # Bounding boxes dictionary
        angle = item.get("angle", 0)  # Rotation angle (default 0)
        org_mask = item.get("mask")  # Original mask image
        
        if img is None or org_mask is None:
            data_dict[image_key]["Mid_Points"] = None
            continue

        # Ensure bboxes is a dictionary containing lists of bounding boxes
        if not isinstance(bboxes, dict):
            data_dict[image_key]["Mid_Points"] = None
            continue

        # Initialize coordinates for Eye and Spina Base
        CoorEye, CoorSb = None, None

        # Extract bounding boxes for Eye and Spina Base
        # y/x ?
        if "Eye" in bboxes:
            CoorEye = (bboxes["Eye"][0]["y_center"], bboxes["Eye"][0]["x_center"])
        if "Spina base" in bboxes:
            CoorSb = (bboxes["Spina base"][0]["y_center"], bboxes["Spina base"][0]["x_center"])
        

        # Ensure both key points exist before proceeding
        if None in (CoorEye, CoorSb):
            data_dict[image_key]["Mid_Points"] = None
            continue

        try:
            # Transform coordinates based on rotation
            Eye_trans = point_trans(CoorEye, np.deg2rad(angle), org_mask.shape, img.shape)
            Sb_trans = point_trans(CoorSb, np.deg2rad(angle), org_mask.shape, img.shape)

            # Calculate midpoint
            MidX = (Eye_trans[0] + Sb_trans[0]) / 2
            MidY = (Eye_trans[1] + Sb_trans[1]) / 2
            Midpoint = (int(MidX), int(MidY))

        except Exception as e:
            print(f"Error processing {image_key}: {e}")
            Midpoint = None  # Set midpoint to None on failure

        # Store midpoint in dictionary
        data_dict[image_key]["Mid_Points"] = Midpoint

        # **Visualize key points on the rotated mask**
        if debug == True:
              output_folder = outputdir +"/transformation"
              os.makedirs(output_folder, exist_ok=True)  # Ensure output directory exists

              # Convert mask to color if grayscale
              if len(img.shape) == 2:
                  img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
              # Draw Eye (Blue), Spina Base (Red), and Midpoint (Green)
              cv2.circle(img, Eye_trans, 5, (255, 0, 0), -1)  # Blue for Eye
              cv2.circle(img, Sb_trans, 5, (0, 0, 255), -1)  # Red for Spina Base
              cv2.circle(img, Midpoint, 5, (0, 255, 0), -1)  # Green for Midpoint
      
              # Draw a line from Eye → Midpoint → Spina Base
              cv2.line(img, Eye_trans, Midpoint, (255, 255, 0), 2)  # Cyan line Eye → Midpoint
              cv2.line(img, Midpoint, Sb_trans, (255, 255, 0), 2)  # Cyan line Midpoint → Spina Base
      
              # Save the visualized image
              output_path = os.path.join(output_folder, f"{image_key}_midpoint_visualized.png")
              cv2.imwrite(output_path, img)
              print(f"Saved midpoint visualization: {output_path}")

    return data_dict  # Return updated dictionary

import cv2
import numpy as np
import os

def Measure_Width_Imhof(data_dict, debug = False, OutputDir = False):
    """
    Measures the width at the detected midpoints for each rotated image,
    stores the results back into data_dict, and visualizes the width measurement.

    :param data_dict: Dictionary containing rotated images and midpoints.
    :param output_folder: Directory to save visualized images.
    :return: Updated data_dict with width measurements added.
    """

    for image_key, item in data_dict.items():
        rotated_image = item.get("rotated_mask")  # Get rotated image
        midpoint = item.get("Mid_Points")  # Get midpoint coordinates
      
        if rotated_image is None or midpoint is None:
            data_dict[image_key]["Width_Measurements"] = {"width": 0, "X_start": 0, "X_end": 0}
            continue  # Skip if missing data

        MidRow = int(midpoint[1])  # Y coordinate of midpoint

        # Ensure MidRow is within valid image bounds
        if MidRow < 0 or MidRow >= rotated_image.shape[0]:
            data_dict[image_key]["Width_Measurements"] = {"width": 0, "X_start": 0, "X_end": 0}
            continue

        # Measure width at the midpoint row
        width = np.sum(rotated_image[MidRow, :]) / 255

        # Find X_start and X_end
        reshaped_row = rotated_image[MidRow, :].reshape(-1, 1)
        X_start = np.argmax(reshaped_row)  # First non-zero pixel
        X_end = len(reshaped_row) - np.argmax(reshaped_row[::-1]) - 1  # Last non-zero pixel

        # Store results in data_dict
        data_dict[image_key]["Width_Measurements"] = {
            "width": width,
            "X_start": X_start,
            "X_end": X_end
        }

        # **Draw the measured width points on the rotated mask**
        
        # Convert mask to color if grayscale
        if len(rotated_image.shape) == 2:
            rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_GRAY2BGR)
        
        if debug == True:
          output_folder = OutputDir +"rotated"
          os.makedirs(output_folder, exist_ok=True)  # Ensure the output directory exists  
          # Draw entry and exit points
          point_color = (0, 255, 0)  # Green
          cv2.circle(rotated_image, (X_start, MidRow), 5, point_color, -1)  # Start point
          cv2.circle(rotated_image, (X_end, MidRow), 5, point_color, -1)  # End point
  
          # Draw a blue line between the points
          line_color = (255, 0, 0)  # Blue
          cv2.line(rotated_image, (X_start, MidRow), (X_end, MidRow), line_color, 2)
  
          # Save the visualized image
          output_path = os.path.join(output_folder, f"{image_key}_width_visualized.png")
          cv2.imwrite(output_path, rotated_image)
          print(f"Saved width visualization: {output_path}")

    return data_dict  # Return updated dictionary


def Measure_Width_Rabus(data_dict, debug = False, output_folder="width_visualizations"):
    """
    Measures the widest row in rotated images and finds the X_start and X_end points.
    Saves visualized masks with detected width points.

    Updates the data_dict with the calculated values.

    :param data_dict: Dictionary containing image data, including rotated masks.
    :param output_folder: Folder to save the width visualization images.
    :return: Updated data_dict with width measurements.
    """
    if debug == True:
      os.makedirs(output_folder, exist_ok=True)  # Ensure output folder exists

    for image_key, data in data_dict.items():
        try:
            rotated_image = data.get("rotated_mask", None)

            if rotated_image is None:
                continue

            # Compute width: Count white pixels per row and find the row with the max width
            row_sums = np.sum(rotated_image == 255, axis=1)  # Count white pixels per row
            max_index = np.argmax(row_sums)  # Row index of the max width
            max_width = row_sums[max_index]  # Actual width in pixels

            # Compute X_start and X_end
            row_values = rotated_image[max_index]  # Get the row with max width
            white_pixels = np.where(row_values == 255)[0]  # Get indices of white pixels

            if white_pixels.size > 0:
                X_start, X_end = white_pixels[0], white_pixels[-1]
            else:
                X_start, X_end = 0, 0

            # Store results in dictionary
            data["Width_Measurements"] = {
                "width": max_width,
                "X_start": X_start,
                "X_end": X_end
            }
            
            if debug == True:
              # Visualize the mask and detected width points
              plt.figure(figsize=(8, 8))
              plt.imshow(rotated_image, cmap="gray")
  
              # Draw detected width line
              plt.plot([X_start, X_end], [max_index, max_index], color="red", linewidth=2)
  
              # Mark endpoints
              plt.scatter([X_start, X_end], [max_index, max_index], color="black", s=40, label="Width Endpoints")
  
              plt.title(f"Width Measurement for {image_key}")
              plt.legend()
  
              # Save the image
              output_path = os.path.join(output_folder, f"{image_key}_width.png")
              plt.savefig(output_path, dpi=300, bbox_inches="tight")
              plt.close()  # Close the figure to prevent memory issues

        except Exception as e:
            print(f"Error processing {image_key}: {e}")
            data["Width_Measurements"] = {
                "width": 0,
                "X_start": 0,
                "X_end": 0
            }

    return data_dict  # Return the updated dictionary

def Create_Visualization_Data(data_dict):
    """
    Translates the measured width coordinates in rotated images back to the original image.

    Updates data_dict with entry and exit points for visualization.

    :param data_dict: Dictionary containing all required data.
    :return: Updated data_dict with visualization coordinates.
    """

    for image_name, data in data_dict.items():
        try:

            # Extract width measurement data properly
            width_data = data.get("Width_Measurements", {})
            X_start = width_data.get("X_start", None)
            X_end = width_data.get("X_end", None)
            
            # Extract other necessary data
            Midpoint = data.get("Mid_Points", None)
            angle = data.get("angle", 0)  # Default to 0 if no rotation
            Rotated_mask = data.get("rotated_mask", None)
            Original_mask = data.get("mask", None)

            # Skip if any essential data is missing
            if X_start is None or X_end is None or Midpoint is None or Rotated_mask is None or Original_mask is None:
                print(f"Warning Create_Visualization_Data: Missing data for {image_name}. Skipping...")
                continue

            EntryY = ExitY = Midpoint[1]  # Y-coordinates remain the same

            if angle != 0:  # If image is rotated, transform coordinates
                EntryX, EntryY = point_trans((X_start, EntryY), np.deg2rad(-angle), 
                                             Rotated_mask.shape, Original_mask.shape)
                ExitX, ExitY = point_trans((X_end, ExitY), np.deg2rad(-angle), 
                                           Rotated_mask.shape, Original_mask.shape)
                                           
            else:  # If no rotation, use the raw values
                EntryX, ExitX = X_start, X_end
                EntryY, ExitY = Midpoint[1], Midpoint[1]  # Y remains the same for non-rotated

            # Store the results back into the dictionary
            data["Width_X1"], data["Width_Y1"] = EntryX, EntryY
            data["Width_X2"], data["Width_Y2"] = ExitX, ExitY

            
        except Exception as e:
            print(f"Error processing {image_name}: {e}")
            data["Width_X1"], data["Width_Y1"] = 0, 0
            data["Width_X2"], data["Width_Y2"] = 0, 0

    return data_dict  # Return the updated dictionary

def Measure_Width_Sperfeld(data_dict, visualize=False):
    """
    Computes the midpoint between the eye and spina base and draws a perpendicular line.
    
    Updates data_dict with:
    - Midpoint coordinates
    - Rotated mask with perpendicular line aligned
    - Rotation angle

    :param data_dict: Dictionary containing image data, including bounding boxes.
    :param visualize: If True, displays visualization of the results.
    :return: Updated data_dict with perpendicular line information.
    """
    for image_key, data in data_dict.items():
        try:
            img = data.get("org_mask", None)
            bboxes = data.get("bboxes", {})
            
            if img is None or "Eye" not in bboxes or "Spina base" not in bboxes:
                print(f"Warning Measure_Width_Sperfeld: Missing data for {image_key}. Skipping...")
                continue
            
            # Extract eye and spina base coordinates
            CoorEye = (bboxes["Eye"][0]["x_center"], bboxes["Eye"][0]["y_center"])
            CoorSb = (bboxes["Spina base"][0]["x_center"], bboxes["Spina base"][0]["y_center"])
            
            # Calculate midpoint
            MidX = (CoorEye[0] + CoorSb[0]) / 2
            MidY = (CoorEye[1] + CoorSb[1]) / 2
            Midpoint = (MidX, MidY)

            # Compute perpendicular line slope
            slope = (CoorSb[1] - CoorEye[1]) / (CoorSb[0] - CoorEye[0])
            perpendicular_slope = -1 / slope

            # Compute line endpoints
            line_length = min(img.shape[1], img.shape[0])
            x_end = Midpoint[0] + line_length / (2 * np.sqrt(1 + perpendicular_slope**2))
            y_end = Midpoint[1] + perpendicular_slope * (x_end - Midpoint[0])

            # Compute rotation angle
            angle = np.arctan(slope)
            rotated_image = imutils.rotate_bound(img, 270 - np.rad2deg(angle))

            # Transform points for rotated image
            angle_c = 1.5 * np.pi - angle
            TransSb = point_trans(CoorSb, angle_c, img.shape, rotated_image.shape)
            TransEye = point_trans(CoorEye, angle_c, img.shape, rotated_image.shape)
            TransMid = point_trans(Midpoint, angle_c, img.shape, rotated_image.shape)
            Line_endpoint = point_trans((x_end, y_end), angle_c, img.shape, rotated_image.shape)

            # Store results in dictionary
            data["Mid_Points"] = Midpoint
            data["Rotated_Mask_Perpendicular"] = rotated_image
            data["Rotation_Angle"] = 270 - np.rad2deg(angle)

            # Visualization (if enabled)
            if visualize:
                plt.clf()
                plt.subplot(1, 2, 2)
                plt.imshow(rotated_image, cmap="gray")
                plt.scatter(TransMid[0], TransMid[1], color='blue', label="Midpoint")
                plt.scatter(TransEye[0], TransEye[1], color='red', label="Eye")
                plt.scatter(TransSb[0], TransSb[1], color='green', label="Spina Base")
                plt.plot([TransMid[0], Line_endpoint[0]], [TransMid[1], Line_endpoint[1]], 'r-', label="Perpendicular Line")
                plt.legend()
                
                plt.subplot(1, 2, 1)
                plt.imshow(img, cmap="gray")
                plt.scatter(Midpoint[0], Midpoint[1], color='blue', label="Midpoint")
                plt.scatter(CoorEye[0], CoorEye[1], color='red', label="Eye")
                plt.scatter(CoorSb[0], CoorSb[1], color='green', label="Spina Base")
                plt.plot([Midpoint[0], x_end], [Midpoint[1], y_end], 'r-', label="Perpendicular Line")
                plt.legend()
                plt.show()

        except Exception as e:
            print(f"Error processing {image_key}: {e}")
            data["Mid_Points"] = None
            data["Rotated_Mask_Perpendicular"] = None
            data["Rotation_Angle"] = None

    return data_dict  # Updated dictionary

def Calculate_Width_At_Midpoint(data_dict):
    """
    Calculates the body width at the midpoint and updates data_dict.

    Updates each entry with:
    - `Width_At_Midpoint`: The calculated width in pixels.
    - `X_start` and `X_end`: The entry and exit X-coordinates of the width.

    :param data_dict: Dictionary containing rotated images and midpoints.
    :return: Updated data_dict with width measurements.
    """
    for image_key, data in data_dict.items():
        try:
            rotated_img = data.get("Rotated_Mask_Perpendicular", None)
            midpoint = data.get("Mid_Points", None)

            if rotated_img is None or midpoint is None:
                print(f"Warning Midpoint_calc: Missing data for {image_key}. Skipping...")
                continue

            MidRow = int(midpoint[1])  # Y-coordinate of the midpoint

            if MidRow < 0 or MidRow >= rotated_img.shape[0]:  # Ensure valid row index
                print(f"Warning: Invalid midpoint Y-coordinate for {image_key}. Skipping...")
                continue

            # Compute width by summing pixels in the selected row
            Width = np.sum(rotated_img[MidRow, :]) / 255  # Normalize by pixel intensity

            # Find X_start and X_end
            reshaped_row = rotated_img[MidRow, :].reshape(-1, 1)
            X_start = np.argmax(reshaped_row)
            X_end = len(reshaped_row) - np.argmax(reshaped_row[::-1]) - 1

            # Store results in dictionary
            data["Width_At_Midpoint"] = Width
            data["Width_X1"] = X_start
            data["Width_X2"] = X_end

        except Exception as e:
            print(f"Error processing {image_key}: {e}")
            data["Width_At_Midpoint"] = 0
            data["Width_X1"] = 0
            data["Width_X2"] = 0

    return data_dict  # Updated dictionary


def WidthImhof(ImageFolder, OutputFolder):
  # Example usage
  test = process_image_folder(ImageFolder,OutputFolder)
  #print(test)
  mask_dir = Path(OutputFolder) / 'Segmentation' / 'mask'
  
  Image_Rotation(test)
  Midpoints = Detect_Midpoint(test)
  Width = Measure_Width_Imhof(Midpoints)
  
  ## Do this for non org images
  
  
  ## This for org images
  Values_To_Be_Drawn = Create_Visualization_Data(Width)

  data = pd.DataFrame.from_dict(Values_To_Be_Drawn,orient='index')
  
  data.to_csv(f"{OutputFolder}/data.csv")
  return(Values_To_Be_Drawn)

# Imhof broades oint straight daphnia (not perpendicular to eye spina axis)
# Sperfeld broades point perpendicular to eye spina
# Rabus Maximum distance between the dorsal and the ventral edge of the carapace

def WidthRabus(ImageFolder, OutputFolder):
  # Example usage
  test = process_image_folder(ImageFolder,OutputFolder)
  mask_dir = Path(OutputFolder) / 'Segmentation' / 'mask'

  #print(test)
  Image_Rotation(test)
  Midpoints = Detect_Midpoint(test)
  Width = Measure_Width_Rabus(Midpoints, "/home/philipp/vis_rabus")
  Values_To_Be_Drawn = Create_Visualization_Data(Width)
  
  return(Values_To_Be_Drawn)

## We take the original Eye position in pixel
## The back rotated width location from Values to be drawn and connect
## thats it
import os
import cv2

def visualize_and_save(data_dict, output_folder):
    """
    Draws lines between Eye → Spina Base and Spina Tip → Spina Base, along with width measurements.
    Saves the annotated images in the specified output folder.

    Parameters:
        data_dict (dict): Dictionary containing image annotations.
        output_folder (str): Path to the folder where visualized images will be saved.
    """
    
    """ Width values are funky """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Define colors
    line_color = (0, 0, 255)  # Red for lines
    dot_color = (0, 0, 0)  # Black for endpoints
    width_color = (0, 255, 0)  # Green for width line

    # Iterate over dictionary entries
    for image, annotations in data_dict.items():

        # Load image
        image_path = annotations.get("image_path")
        if not image_path or not os.path.exists(image_path):
            print(f"Warning: Image not found for {image}")
            continue
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            continue

        img_height, img_width = img.shape[:2]  # Get image dimensions

        # Extract keypoints in pixel coordinates
        keypoints = {}
        for key in ["Eye", "Spina base", "Spina tip"]:
            if key in annotations.get("bboxes", {}):
                box = annotations["bboxes"][key][0]  # Assuming one box per key
                x_center, y_center = box["x_center"], box["y_center"]
                keypoints[key] = (int(x_center), int(y_center))  # Already in pixels

        # Draw lines between keypoints
        connections = [("Eye", "Spina base"), ("Spina tip", "Spina base")]
        for pt1, pt2 in connections:
            if pt1 in keypoints and pt2 in keypoints:
                cv2.line(img, keypoints[pt1], keypoints[pt2], line_color, 3, cv2.LINE_AA)
                cv2.circle(img, keypoints[pt1], 5, dot_color, -1)
                cv2.circle(img, keypoints[pt2], 5, dot_color, -1)

        try:
          x1 = int(annotations.get("Width_Y1"))
          y1 = int(annotations.get("Width_X1"))
          x2 = int(annotations.get("Width_Y2"))
          y2 = int(annotations.get("Width_X2"))
  
          # Draw width line
          cv2.line(img, (x1, y1), (x2, y2), width_color, 2, cv2.LINE_AA)
          cv2.circle(img, (x1, y1), 5, dot_color, -1)
          cv2.circle(img, (x2, y2), 5, dot_color, -1)
        except:
          print(f"no measurement for {image}")


        # Save the visualized image
        output_path = os.path.join(output_folder, image)
        cv2.imwrite(output_path, img)
        print(f"Saved at {output_path}")
        

