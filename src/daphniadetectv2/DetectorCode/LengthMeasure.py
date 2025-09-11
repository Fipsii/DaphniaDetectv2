import numpy as np
import pandas as pd
def MeasureLength(data_dict):
    """
    Measures distances between key anatomical points when data is available:
    - Eye → Spina Base
    - Spina Tip → Spina Base
    """
    for image_key, annotations in data_dict.items():
        # Extract center points of objects
        centers = {}
        for key in ["Eye", "Spina base", "Spina tip"]:
            box = annotations.get("bboxes", {}).get(key)
            if box:
                centers[key] = (box[0]["x_center"], box[0]["y_center"])  # Assuming only one bbox
            else:
                centers[key] = None

        distances = {}

        # Compute Eye to Spina Base if both exist
        if centers["Eye"] and centers["Spina base"]:
            distances["Eye_to_SpinaBase"] = np.linalg.norm(np.array(centers["Eye"]) - np.array(centers["Spina base"]))

        # Compute Spina Tip to Spina Base if both exist
        if centers["Spina tip"] and centers["Spina base"]:
            distances["SpinaTip_to_SpinaBase"] = np.linalg.norm(np.array(centers["Spina tip"]) - np.array(centers["Spina base"]))

        if distances:
            data_dict[image_key]["Distances"] = distances
        else:
            
            data_dict[image_key]["Distances"] = None

    return data_dict


  
def scale_values(row):
    
    image_name = row["image_name"]
    distance_dict = row["Distances"] if isinstance(row["Distances"], dict) else {}
    width_dict = row["Width_Measurements"] if isinstance(row["Width_Measurements"], dict) else {}
    scale_factor = row["distance_per_pixel"] if not pd.isna(row["distance_per_pixel"]) else 1  # Default to 1 if NaN

    # Scale values safely, using `.get()` to avoid KeyErrors and handling NaNs
    return {
        "image_name": image_name,
        "Eye-Base_µm": (distance_dict.get("Eye_to_SpinaBase", np.nan) * scale_factor) if distance_dict.get("Eye_to_SpinaBase") is not None else np.nan,
        "Base-Tip_µm": (distance_dict.get("SpinaTip_to_SpinaBase", np.nan) * scale_factor) if distance_dict.get("SpinaTip_to_SpinaBase") is not None else np.nan,
        "Width_µm": (width_dict.get("width", np.nan) * scale_factor) if width_dict.get("width") is not None else np.nan,
        "X_start": width_dict.get("X_start", np.nan),  # Keep as is
        "X_end": width_dict.get("X_end", np.nan),  # Keep as is
        "distance_per_pixel": scale_factor  # Store distance scaling factor
    }
