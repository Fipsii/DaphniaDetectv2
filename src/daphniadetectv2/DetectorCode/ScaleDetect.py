import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

import numpy as np
import cv2
import re
import easyocr
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt

READER = easyocr.Reader(['en'])

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def safe_to_gray(image):
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def parse_to_mm(text):
    match = re.search(r'([-+]?\d*\.?\d+)\s*([a-zA-Zµ]*)', text.strip(), re.IGNORECASE)
    if not match:
        return None, None
        
    val = float(match.group(1))
    raw_unit = match.group(2).lower()

    um_noise = ['um', 'µm', 'pm', 'nn', 'un', 'yn', 'p', 'u', 'µ']
    mm_noise = ['mm', 'nm', 'mn', 'm']

    if any(variant == raw_unit for variant in um_noise): return val / 1000.0, 'µm'
    if any(variant == raw_unit for variant in mm_noise): return val, 'mm'
    return (val / 1000.0, 'µm') if val >= 15.0 else (val, 'mm')

# ---------------------------------------------------------------------------
# Monotone Cluster Topology Replacement Logic
# ---------------------------------------------------------------------------


def find_scale_line_near_text(image_strip, text_y_min, text_y_max, text_x_min, text_x_max):
    gray = safe_to_gray(image_strip)
    h, w = gray.shape
    
    # ---------------------------------------------------------
    # PHASE 1: Attempt Structural Box Detection
    # ---------------------------------------------------------
    pad = 5
    c_y1, c_y2 = max(0, int(text_y_min) - pad), min(h, int(text_y_max) + pad)
    c_x1, c_x2 = max(0, int(text_x_min) - pad), min(w, int(text_x_max) + pad)
    
    text_roi = gray[c_y1:c_y2, c_x1:c_x2]
    if text_roi.size == 0: return None
    
    vals, counts = np.unique(text_roi, return_counts=True)
    target_color = int(vals[np.argmax(counts)])
    
    tolerance = 15
    mask = cv2.inRange(gray, max(0, target_color - tolerance), min(255, target_color + tolerance))
    
    stroke_width = 7 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (stroke_width, stroke_width))
    solid_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(solid_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    valid_clusters = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (text_x_max - text_x_min) * (text_y_max - text_y_min) * 0.5:
            continue 
            
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        if (area / float(cw * ch)) < 0.70:
            continue
            
        overlap_x = max(0, min(cx + cw, text_x_max) - max(cx, text_x_min))
        overlap_y = max(0, min(cy + ch, text_y_max) - max(cy, text_y_min))
        
        if overlap_x * overlap_y > 0:
            valid_clusters.append({'bbox': (cx, cy, cw, ch), 'overlap': overlap_x * overlap_y})

    # ---------------------------------------------------------
    # PHASE 2: Resolve Search Space & Extract Ink 
    # ---------------------------------------------------------
    if valid_clusters:
        # OPTION A: Valid box found. Restrict search to its boundaries.
        valid_clusters.sort(key=lambda x: x['overlap'], reverse=True)
        tx, ty, tw, th = valid_clusters[0]['bbox']
        
        roi_gray = gray[ty:ty+th, tx:tx+tw]
        
        # Ink is anything deviating from the box's background color
        ink_mask = (np.abs(roi_gray.astype(int) - target_color) > 30).astype(np.int8)
        
        # ---------------------------------------------------------
        # PHASE 3: Exact 1D Vectorized Line Extraction
        # ---------------------------------------------------------
        padded = np.pad(ink_mask, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        diff = np.diff(padded, axis=1)
        
        starts = np.where(diff == 1)
        ends = np.where(diff == -1)
        
        if len(starts[0]) == 0:
            return None
            
        lengths = ends[1] - starts[1]
        best_idx = np.argmax(lengths)
    
        # Reject noise blobs smaller than expected scale lines
        min_required_length = max(20, (text_x_max - text_x_min) * 0.4)
        if lengths[best_idx] < min_required_length:
            return None
            
        # Map array indices back to global image coordinates
        absolute_y = ty + starts[0][best_idx]
        absolute_x1 = tx + starts[1][best_idx]
        absolute_x2 = tx + ends[1][best_idx]
        
        return (int(absolute_x1), int(absolute_y), int(absolute_x2), int(absolute_y))
    else:
        # OPTION B: Line Extraction via Horizontal Edge Bridging
        tx, ty = 0, 0
        
        # 1. Canny edges
        edges = cv2.Canny(gray, 40, 150)
        
        # 2. Bridge edges vertically to create a solid 1D mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        solid_line_mask = cv2.dilate(edges, kernel)
        
        # 3. Suppress text regions
        pad_y = 2
        erase_y1 = max(0, int(text_y_min) - pad_y)
        erase_y2 = min(h, int(text_y_max) + pad_y)
        solid_line_mask[erase_y1:erase_y2, :] = 0
        
        # 4. Perform the 1D vectorized scan
        bool_mask = (solid_line_mask > 0).astype(np.int8)
        padded = np.pad(bool_mask, ((0, 0), (1, 1)), mode='constant', constant_values=0)
        diff = np.diff(padded, axis=1)
        
        starts = np.where(diff == 1)
        ends = np.where(diff == -1)
        
        # Find longest run
        lengths = ends[1] - starts[1]
        best_idx = np.argmax(lengths)
        
        best_y = starts[0][best_idx]
        best_x1 = starts[1][best_idx]
        best_x2 = ends[1][best_idx]
        
    # Map back to global strip coordinates
    absolute_y = ty + best_y
    absolute_x1 = tx + best_x1
    absolute_x2 = tx + ends[1][best_idx] # ensured sync with best_idx
    
    return (int(absolute_x1), int(absolute_y), int(absolute_x2), int(absolute_y))
   

# ---------------------------------------------------------------------------
# Execution Pipeline Execution Flow
# ---------------------------------------------------------------------------

def process_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    h, w = image.shape[:2]
    start_y, start_x = h // 2, w // 2
    roi = image[start_y:, start_x:]

    gray_roi = safe_to_gray(roi)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_roi = clahe.apply(gray_roi)

    ocr_results = READER.readtext(
        enhanced_roi, 
        allowlist='0123456789.µumpn ', 
        text_threshold=0.05, 
        low_text=0.1,        
        width_ths=1.0,       
        mag_ratio=2.0,       
        contrast_ths=0.05,   
        min_size=5 
    )

    candidates = []
    for bbox, text, conf in ocr_results:
        if any(c.isdigit() for c in text):
            candidates.append((bbox, text.strip(), conf))

    if not candidates:
        return {"error": "No numeric scale text detected by OCR"}

    candidates.sort(key=lambda x: x[2], reverse=True)
    best_bbox, best_text, _ = candidates[0]
    sanitized_text = best_text.translate(str.maketrans('Il|OoQDSGbBUM', '11100005566um'))
    sanitized_text = sanitized_text.replace("6m", "00").replace("6M", "00")
    metric_length_mm, unit = parse_to_mm(sanitized_text)

    
    # 2. ANCHOR & SEARCH: Look for the line near the text
    y_coords = [pt[1] for pt in best_bbox]
    x_coords = [pt[0] for pt in best_bbox]
    
    text_y_min, text_y_max = int(min(y_coords)), int(max(y_coords))
    text_x_min, text_x_max = int(min(x_coords)), int(max(x_coords))

    # Keep Y tight to prevent vertical noise, but take the FULL width for X
    strip_y_min = max(0, text_y_min - 40)
    strip_y_max = min(roi.shape[0], text_y_max + 40)
    
    strip_x_min = 0
    strip_x_max = roi.shape[1]

    # Calculate text coordinates relative to the new cropped strip
    local_text_y_min = text_y_min - strip_y_min
    local_text_y_max = text_y_max - strip_y_min
    local_text_x_min = text_x_min - strip_x_min
    local_text_x_max = text_x_max - strip_x_min


    search_strip = roi[strip_y_min:strip_y_max, strip_x_min:strip_x_max]
    
    line_segment = find_scale_line_near_text(
        search_strip, 
        local_text_y_min, 
        local_text_y_max, 
        local_text_x_min, 
        local_text_x_max
    )
    
    if not line_segment:
        return {"error": "Text found, but no matching scale structure discovered"}

    lx1, ly, lx2, _ = line_segment
    
    global_x1 = lx1 + strip_x_min + start_x
    global_x2 = lx2 + strip_x_min + start_x
    global_y = ly + strip_y_min + start_y

    scale_px = abs(global_x2 - global_x1)
    distance_per_pixel = metric_length_mm / scale_px if metric_length_mm and scale_px else None

    return {
        "raw_text": sanitized_text,
        "metric_length_mm": metric_length_mm,
        "scale_px": scale_px,
        "distance_per_pixel": distance_per_pixel,
        "global_line_coords": (global_x1, global_y, global_x2, global_y)
    }

# ---------------------------------------------------------------------------
# Data Frame Processing & Visualization Functions
# ---------------------------------------------------------------------------

def makeDfwithfactors(list_of_names, ConvFactor, Scale_Mode, Values=[], Lines=[]):
    list_of_lengths = [abs(l[2] - l[0]) if l is not None else None for l in Lines]
    list_of_mm = [v[0] if v is not None else None for v in Values]

    if Scale_Mode == 0:
        df = pd.DataFrame({'image_name': list_of_names})
        df["distance_per_pixel"] = ConvFactor
        return df

    df = pd.DataFrame({
        'image_name': list_of_names,
        'metric_length': list_of_mm,
        'scale[px]': list_of_lengths,
        'coordinates_scale': Lines
    })

    if Scale_Mode == 1:
        mode_px = df['scale[px]'].mode()[0] if not df['scale[px]'].mode().empty else None
        mode_mm = df['metric_length'].mode()[0] if not df['metric_length'].mode().empty else None
        df['scale[px]'] = df['scale[px]'].fillna(mode_px)
        df['metric_length'] = df['metric_length'].fillna(mode_mm)

    df["distance_per_pixel"] = df["metric_length"] / df["scale[px]"]
    return df

def Scale_Measure(DataDict, Scale_Mode, Conv_factor=0.002139):
    names = list(DataDict.keys())
    if Scale_Mode == 0:
        return makeDfwithfactors(names, Conv_factor, Scale_Mode)

    results_data = []
    lines_data = []
    
    for name in names:
        path = DataDict[name].get("image_path")
        try:
            res = process_single_image(path)
            if "error" not in res:
                results_data.append((res["metric_length_mm"], "mm"))
                lines_data.append(res["global_line_coords"])
            else:
                results_data.append(None)
                lines_data.append(None)
        except Exception:
            results_data.append(None)
            lines_data.append(None)
            
    return makeDfwithfactors(names, Conv_factor, Scale_Mode, results_data, lines_data)

def visualize_measurements(data_dict, results_df, output_folder="annotated_images"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    for index, row in results_df.iterrows():
        file_name = row['image_name'] 
        img_path = data_dict[file_name]["image_path"]
        coords = row['coordinates_scale'] 
        
        if coords is not None:
            img = cv2.imread(img_path)
            cv2.line(img, (int(coords[0]), int(coords[1])), 
                          (int(coords[2]), int(coords[3])), (0, 255, 0), 2)
            save_path = os.path.join(output_folder, f"annotated_{file_name}")
            cv2.imwrite(save_path, img)
            print(f"Saved Target Vector View: {save_path}")

# ---------------------------------------------------------------------------
# Initializer Loop Execution Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    folder_path = r"C:\Users\hanss\Desktop\Daphnia\2025_Images_Sorted\Test_Images\ScalesTest"
    
    image_files = glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.png")) + \
                  glob.glob(os.path.join(folder_path, "*.png"))

    data_dict = {os.path.basename(f): {"image_path": f} for f in image_files}
    print(f"Total images found: {len(data_dict)}")

    if len(data_dict) > 0:
        results_df = Scale_Measure(data_dict, Scale_Mode=2)
        visualize_measurements(data_dict, results_df)
    else:
        print("No image data extracted.")