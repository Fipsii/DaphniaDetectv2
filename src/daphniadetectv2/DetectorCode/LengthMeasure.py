import numpy as np
import pandas as pd
def MeasureLength(data_dict):
    """
    Measures the distances between:
    - Eye → Spina Base
    - Spina Tip → Spina Base

    Uses the center coordinates from bounding boxes.

    Parameters:
        data_dict (dict): Dictionary containing image annotations, including bounding box center points.

    Returns:
        dict: Updated data_dict with distances added under ["Distances"].
    """
    for image_key, annotations in data_dict.items():
        # Extract center points of objects
        centers = {key: None for key in ["Eye", "Spina base", "Spina tip"]}

        for key in centers.keys():
            if key in annotations.get("bboxes", {}):
                box = annotations["bboxes"][key][0]  # Assuming one bounding box per key
                centers[key] = (box["x_center"], box["y_center"])  # Center coordinates

        # Ensure all required keypoints exist
        if None in centers.values():
            print(f"Skipping {image_key}: Missing keypoints")
            data_dict[image_key]["Distances"] = None
            continue

        # Compute Euclidean distances
        eye_to_spina_base = np.linalg.norm(np.array(centers["Eye"]) - np.array(centers["Spina base"]))
        spina_tip_to_spina_base = np.linalg.norm(np.array(centers["Spina tip"]) - np.array(centers["Spina base"]))

        # Store distances in the dictionary
        data_dict[image_key]["Distances"] = {
            "Eye_to_SpinaBase": eye_to_spina_base,
            "SpinaTip_to_SpinaBase": spina_tip_to_spina_base
        }

    return data_dict  # Return updated dictionary with distances

def scale_values(row):
    distance_dict = ast.literal_eval(row["Distances"])  # Convert string to dictionary
    width_dict = ast.literal_eval(row["Width_Measurements"])  # Convert string to dictionary
    scale_factor = row["distance_per_pixel"]

    # Scale numerical values (excluding "X_start" and "X_end")
    scaled_distance = {key + "_mm": value * scale_factor for key, value in distance_dict.items()}
    scaled_width = {key + "_mm": (value * scale_factor if key != "X_start" and key != "X_end" else value) for key, value in width_dict.items()}

    return {**scaled_distance, **scaled_width, "distance_per_pixel": scale_factor}

  
def scale_values(row):
    distance_dict = row["Distances"] if isinstance(row["Distances"], dict) else {}
    width_dict = row["Width_Measurements"] if isinstance(row["Width_Measurements"], dict) else {}
    scale_factor = row["distance_per_pixel"] if not pd.isna(row["distance_per_pixel"]) else 1  # Default to 1 if NaN

    # Scale values safely, using `.get()` to avoid KeyErrors and handling NaNs
    return {
        "Eye-Base_mm": (distance_dict.get("Eye_to_SpinaBase", np.nan) * scale_factor) if distance_dict.get("Eye_to_SpinaBase") is not None else np.nan,
        "Base-Tip_mm": (distance_dict.get("SpinaTip_to_SpinaBase", np.nan) * scale_factor) if distance_dict.get("SpinaTip_to_SpinaBase") is not None else np.nan,
        "Width_mm": (width_dict.get("width", np.nan) * scale_factor) if width_dict.get("width") is not None else np.nan,
        "X_start": width_dict.get("X_start", np.nan),  # Keep as is
        "X_end": width_dict.get("X_end", np.nan),  # Keep as is
        "distance_per_pixel": scale_factor  # Store distance scaling factor
    }
