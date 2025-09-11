import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import easyocr
import cv2
import matplotlib.pyplot as plt
import re
import pandas as pd
import time
def safe_to_gray(image):

    try:

        # If it's already grayscale, this might raise an error or have no effect
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        # If conversion fails, assume it's already grayscale or invalid
        if len(image.shape) == 2:
            gray = image  # Already grayscale
        else:
            raise ValueError(f"Cannot convert image to grayscale: {e}")
    return gray

def find_contrasting_horizontal_line_in_monochrome_box(box_crop, visualize_randomized=True, tolerance=5):
    gray = safe_to_gray(box_crop)
    
    # Find most common pixel value (background)
    vals, counts = np.unique(gray, return_counts=True)

    background_val = vals[np.argmax(counts)]

    # Randomize background pixels
    gray_mod = gray.copy()
    background_mask = (gray == background_val)
    num_bg_pixels = np.sum(background_mask)
    gray_mod[background_mask] = np.random.randint(0, 256, size=num_bg_pixels)

    #if visualize_randomized:
        #plt.figure(figsize=(6,4))
        #plt.title("Randomized Background Pixels in Grayscale Box")
        #plt.imshow(gray_mod, cmap='gray')
        #plt.axis('off')
        #plt.show()
    # Find longest horizontal run of pixels with values within tolerance
    longest_y = None
    longest_len = 0
    longest_segment = (0, 0)
    longest_val = None

    for y in range(gray_mod.shape[0]):
        row = gray_mod[y]
        start = 0
        length = 1
        for i in range(1, len(row)):
            if abs(int(row[i]) - int(row[i-1])) <= tolerance:
                length += 1
            else:
                if length > longest_len:
                    longest_len = length
                    longest_y = y
                    longest_segment = (start, i - 1)
                    longest_val = row[i-1]
                start = i
                length = 1
        # Check last run in row
        if length > longest_len:
            longest_len = length
            longest_y = y
            longest_segment = (start, len(row) - 1)
            longest_val = row[-1]

    if longest_y is not None:
        x1, x2 = longest_segment
        y = longest_y
        
        return (x1, y, x2, y)
    else:
        return None


def expand_edges_by_one_pixel(mask):
    # mask is binary: black edges (0), white background (255)
    h, w = mask.shape
    new_mask = mask.copy()
    
    for y in range(1, h-1):
        for x in range(1, w-1):
            if mask[y, x] == 255:  # white pixel (background)
                # Check 8 neighbors for black pixel (edge)
                neighbors = mask[y-1:y+2, x-1:x+2]
                if np.any(neighbors == 0):
                    new_mask[y, x] = 0  # turn this white pixel black
    
    return new_mask


def find_scale_box_edges(img, color_threshold=15, min_box_fraction=0.05):
    # --- Check image ---
    if img is None:
        print("[ERROR] Image is None")
        return [None, None]

    h, w = img.shape[:2]
    min_box_size_w = int(w * min_box_fraction)

    # --- Helper functions ---
    def is_black(pixel):
        return np.all(pixel < color_threshold)

    def find_edge(img, start, step):
        x, y = start
        while 0 <= x + step[0] < img.shape[1] and 0 <= y + step[1] < img.shape[0]:
            next_pixel = img[y + step[1], x + step[0]]
            if not is_black(next_pixel):
                break
            x += step[0]
            y += step[1]
        return [x, y]

    # --- Precompute masks ---
    if img.ndim == 3:
        black_mask = np.all(img[:, :, :3] < color_threshold, axis=2)
    else:
        black_mask = img < color_threshold

    visited = np.zeros((h, w), dtype=bool)
    boxes = []

    # --- Scan only black+unvisited pixels ---
    ys, xs = np.where(black_mask & ~visited)
    for y, x in zip(ys, xs):
        if visited[y, x]:
            continue

        # Found black pixel → find edges
        top_left = find_edge(img, [x, y], [-1, 0])
        top_right = find_edge(img, [x, y], [1, 0])
        bottom_right = find_edge(img, top_right, [0, 1])
        bottom_left = [top_left[0], bottom_right[1]]

        width = abs(top_right[0] - top_left[0])
        height = abs(bottom_right[1] - top_right[1])

        if width >= min_box_size_w:
            boxes.append([top_left, bottom_right])
            visited[top_left[1]:bottom_right[1]+1,
                    top_left[0]:bottom_right[0]+1] = True

    if not boxes:
        return [None, None]

    # --- Pick the largest box ---
    largest_box = max(boxes, key=lambda b: (b[1][0]-b[0][0]) * (b[1][1]-b[0][1]))

    # --- Visualization ---
    vis = img.copy()
    for box in boxes:
        cv2.rectangle(vis, tuple(box[0]), tuple(box[1]), (255, 0, 0), 2)
    cv2.rectangle(vis, tuple(largest_box[0]), tuple(largest_box[1]), (0, 255, 0), 3)

    return largest_box






import cv2
import numpy as np

def find_rectangles(binary_image, min_area=2000, max_aspect_ratio=5, min_solidity=0.8):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < min_solidity:
            continue
        
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue
        
        aspect_ratio = max(width, height) / min(width, height)
        if aspect_ratio > max_aspect_ratio:
            continue

        # Get rectangle corners
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Convert to top-left and bottom-right
        x_coords = [p[0] for p in box]
        y_coords = [p[1] for p in box]
        top_left = [min(x_coords), min(y_coords)]
        bottom_right = [max(x_coords), max(y_coords)]

        return [top_left, bottom_right]  # Return only the first valid rectangle

    return [None, None]  # If no valid rectangle found


import numpy as np

def shrink_box_to_uniform_border(image, top_left, bottom_right, max_shrink=20, color_tolerance=5):
    def average_color(pixels):
        return np.mean(pixels.reshape(-1, 3), axis=0)

    def edges_are_close(colors, tol):
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                if not np.all(np.abs(colors[i] - colors[j]) <= tol):
                    return False
        return True

    x1, y1 = top_left
    x2, y2 = bottom_right
    shrink_count = 0

    while shrink_count < max_shrink and (x2 - x1 > 10) and (y2 - y1 > 10):
        # Extract edge strips
        top = image[y1, x1:x2]
        bottom = image[y2 - 1, x1:x2]
        left = image[y1:y2, x1]
        right = image[y1:y2, x2 - 1]

        edge_colors = [
            average_color(top),
            average_color(bottom),
            average_color(left),
            average_color(right)
        ]

        # Stop only when edges become similar
        if edges_are_close(edge_colors, color_tolerance):
            break

        # Shrink inward
        x1 += 1
        y1 += 1
        x2 -= 1
        y2 -= 1
        shrink_count += 1

    return [x1, y1], [x2, y2]

def find_strictly_horizontal_line(edges, min_length_ratio=0.05, min_length_fixed=10):
    height, width = edges.shape
    max_len = 0
    best_y = None
    best_segment = (0, 0)

    # Calculate minimum length threshold
    min_length = max(int(width * min_length_ratio), min_length_fixed)

    for y in range(height):
        row = edges[y, :]
        xs = np.where(row > 0)[0]

        if len(xs) == 0:
            continue

        # Find longest continuous run
        start = xs[0]
        prev = xs[0]
        longest_start, longest_end = start, start
        current_start = start

        for x in xs[1:]:
            if x == prev + 1:
                prev = x
            else:
                if prev - current_start + 1 > longest_end - longest_start + 1:
                    longest_start, longest_end = current_start, prev
                current_start = x
                prev = x

        # Final segment
        if prev - current_start + 1 > longest_end - longest_start + 1:
            longest_start, longest_end = current_start, prev

        length = longest_end - longest_start + 1
        if length > max_len:
            max_len = length
            best_y = y
            best_segment = (longest_start, longest_end)

    if best_y is not None and max_len >= min_length:
        return (best_segment[0], best_y, best_segment[1], best_y)
    return None




def detect_lines(image):
    top_left, bottom_right = find_scale_box_edges(image)

    if top_left is not None and bottom_right is not None and not all(abs(a - b) <= 20 for a, b in zip(top_left, bottom_right)):

        top_left, bottom_right = shrink_box_to_uniform_border(image, top_left, bottom_right)

        x1, y1 = top_left
        x2, y2 = bottom_right

        box_crop = image[y1:y2, x1:x2]

        # Find the contrasting horizontal line inside the box
        line_coords = find_contrasting_horizontal_line_in_monochrome_box(box_crop)

        if line_coords is not None:
            # Adjust line coordinates relative to original image
            lx1, ly, lx2, _ = line_coords
            ly += y1
            lx1 += x1
            lx2 += x1
            adjusted_line_coords = (lx1, ly, lx2, ly)
            return (box_crop, adjusted_line_coords)
        else:
            # If line not found in scale box, treat as failure
            return (None, None)

    else:
        # Fallback: try to detect horizontal line directly
        gray = safe_to_gray(image)
        edges = cv2.Canny(gray, 50, 150)
        ##plt.clf()
        ##plt.imshow(edges, cmap='gray')
        ##plt.show()
        line_coords = find_strictly_horizontal_line(edges)

        if line_coords is not None:
            # Try cropping around the detected line
            box_crop = crop_text_around_scale_bar(image, line_coords)
            if box_crop is not None:
                return (box_crop, line_coords)
    
    # If nothing worked, return None
    return (None, None)


	    



#### Now we need to detect the Number
#### Two functions: IF BOX TRUE -> just search in the box
#### If not search in the area above and below the detected line


def Detect(box_crop):
	results = []

	for item in box_crop:
		if item [0] is not None:
			crop_reworked = scale_for_easyocr(item[0])
			#plt.figure(figsize=(6, 6))
			#plt.imshow(crop_reworked, cmap='gray')  # Use 'gray' if the image is grayscale
			#plt.axis('off')
			#plt.show()
			reader = easyocr.Reader(['en'])
			result = reader.readtext(crop_reworked, allowlist='0123456789.µuUmMnNIi')
			results.append(result)
		else:
			results.append(None)

	return results


def crop_text_around_scale_bar(image, bar_coords):
    """
    Crop the box expanded vertically by 10% image height (above and below),
    then detect text inside that crop using EasyOCR.

    Args:
        image_path (str): Path to image.
        box_coords (tuple): (xmin, ymin, xmax, ymax) of original scale bar box.
    """

    reader = easyocr.Reader(['en'])
    h, w, _ = image.shape
    
    if bar_coords is not None:
	    xmin, ymin, xmax, ymax = bar_coords

	    pad = int(h * 0.05)  # 10% padding of image height

	    # Expanded box coordinates
	    y_top = max(ymin - pad, 0)
	    y_bottom = min(ymax + pad, h)

	    # Crop expanded box
	    crop = image[y_top:y_bottom, xmin:xmax]
    else:
            crop = None
    return crop

def scale_for_easyocr(crop):

    # Apply CLAHE
    # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    # enhanced = clahe.apply(gray)
    # Resize

    scale_factor = 4
    resized = cv2.resize(crop, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    #plt.clf()
    #plt.imshow(resized)
    #plt.show()
    return resized


import cv2
import os

def process_folder(List_of_paths):
    import cv2
    import numpy as np

    all_crops = []
    all_results = []

    for path in List_of_paths:
        image = cv2.imread(path)

        if image is None:
            print(f"[WARNING] Could not load image: {path}")
            all_crops.append((None, None))
            all_results.append("NA")
            continue

        crop = detect_lines(image)
        if crop and all(part is not None for part in crop):
            try:
                gray = safe_to_gray(crop[0])
                crop_data = (gray, crop[1])
                all_crops.append(crop_data)
                all_results.append(None)  # Placeholder for detection result
            except Exception as e:
                print(f"[ERROR] Processing failed for {path}: {e}")
                all_crops.append((None, None))
                all_results.append("NA")
        else:
            all_crops.append((None, None))
            all_results.append("NA")

    # Run detection on all valid crops
    valid_crops = [c for c in all_crops if c[0] is not None and c[1] is not None]
    if valid_crops:
        valid_results = Detect(valid_crops)

        result_idx = 0
        for i in range(len(all_results)):
            if all_results[i] is None:  # Placeholder detected earlier
                all_results[i] = valid_results[result_idx]
                result_idx += 1
   
    return all_results, all_crops



    
def lengths(lines):
    lengths = []
    
    for line in lines:
        try:
            x1, y1, x2, y2 = line
            xmin = min(x1, x2)
            xmax = max(x1, x2)
            length = xmax - xmin
            lengths.append(length)
        except:
            lengths.append("NA")
    return lengths

 

def parse_and_convert_to_mm(measurements):
    values_mm = []
    units = []
    pattern = re.compile(r'^([-+]?\d*\.?\d+)\s*(mm|um|µm|m|nm|nn|n)?$', re.IGNORECASE)
    
    for item in measurements:
        item = item.strip()

        match = pattern.match(item)

        if match:
            value = float(match.group(1))
            unit = match.group(2).lower() if match.group(2) else None
       
            if unit in ['um', 'µm']:
                value_mm = value / 1000.0
                unit_final = unit
            elif unit == 'mm':
                value_mm = value
                unit_final = unit
            elif value > 15:
                # Assume mm if unit is missing and value > 10
                value_mm = value / 1000.0
                unit_final = 'mm'
            elif value < 15:
                # Assume mm if unit is missing and value > 10
                value_mm = value
                unit_final = 'µm'
            else:
                value_mm = None
                unit_final = None
        else:
            value_mm = None
            unit_final = None

        values_mm.append(value_mm)
        units.append(unit_final)
    
    return values_mm, units




def safe_int_list(values):
    out = []
    for v in values:
        try:
            if v is None or (isinstance(v, str) and v.strip().upper() == "NA"):
                out.append(np.nan)   # mark invalid as NaN
            else:
                out.append(int(v))
        except Exception:
            out.append(np.nan)
    return out

def safe_float_list(values):
    out = []
    for v in values:
        try:
            if v is None or (isinstance(v, str) and v.strip().upper() == "NA"):
                out.append(np.nan)
            else:
                out.append(float(v))
        except Exception:
            out.append(np.nan)
    return out

def makeDfwithfactors(list_of_names,ConvFactor, Scale_Mode,Values=[],Lines =[]):
  
  ### This function has two modes. 1) If the user declares that we only have one 
  ### scale we take the most common values of length and unit and 2) if more 
  ### than one exist we keep the list as they are.
  ### Then we enter the singular or mutliple values into the df
  
  import pandas as pd
  
  
  list_of_lengths =  lengths(Lines)

  List_of_scale_numbers, List_of_scale_units = parse_and_convert_to_mm(Values)
  
  if Scale_Mode == 0:
    print(f"Using manual factor of {ConvFactor} px/mm")
    LengthOpt = 0
    UnitOpt = 0
    Scale_df = pd.DataFrame({'image_name': list_of_names})
    Scale_df["distance_per_pixel"] = ConvFactor
    
    return Scale_df
  
  # Uniform Scale

  elif Scale_Mode == 1:

    # Filter out invalid entries
    valid_lengths = [x for x in list_of_lengths if x != "NA"]
    # Compute LengthOpt safely
    if valid_lengths:
     LengthOpt = int(max(set(valid_lengths), key=valid_lengths.count))
    else:
     LengthOpt = None  # or a default value like 0
     
     # Similarly for UnitOpt
     valid_units = [x for x in List_of_scale_numbers if x not in ("NA", None)]
     if valid_units:
      UnitOpt = float(max(set(valid_units), key=valid_units.count))
     else:
      UnitOpt = None  # or default like 1.0


  # Different Scales
  elif Scale_Mode == 2:

    try:
     LengthOpt = safe_int_list(list_of_lengths)
     UnitOpt   = safe_float_list(List_of_scale_numbers)
     
     ## We change unusual scale values that can be confused with 1
     UnitOpt = [1.0 if x == 7.0 else x for x in UnitOpt]
     UnitOpt = [1.0 if x == 4.0 else x for x in UnitOpt]
    except:
    	print("Error in makeDfwithfactors")
  else:
    print("No mode detected")  

  ## Note we add the individual lines to the scale mode who expects unifrom
  ## scales. This allows the user to see the stability of detection, but
  ## could cause confusion if looking at results.
  
  Scale_df = pd.DataFrame({'image_name': list_of_names})
  
  Scale_df["metric_length"] = UnitOpt
  Scale_df["scale[px]"] = LengthOpt
  Scale_df["coordinates_scale"] = Lines
  Scale_df["distance_per_pixel"] = Scale_df["metric_length"]/Scale_df["scale[px]"]
    
  return Scale_df
  




def Scale_Measure(DataDict, Scale_Mode, Conv_factor=0.002139):
    Paths_of_Images = []
    Name_of_Images = []

    for image, annotations in DataDict.items():
        image_path = annotations.get("image_path")
        Paths_of_Images.append(image_path)
        Name_of_Images.append(image)

    # Wrap processing in progress bar

    results = []
    crops = []

    # Use tqdm to show a single filling bar
    for image_path in tqdm(Paths_of_Images, desc="Detecting scales", unit="img"):
        # Assuming process_folder takes a list of paths
        # So we call it on [image_path] to process one at a time
        

        res, crp = process_folder([image_path])


        
        results.append(res)
        crops.append(crp[0] if crp else None)

    if Scale_Mode != 0:
        # Flatten all tuples from all sublists inside results
        flattened = [item for sublist in results for item in sublist]
        text = []

        for res in results:
            if res is None:
                text.append("NA")
            else:
                try:
                    _, txt, _ = res[0]  # assuming res is a list of tuples
                    text.append(str(txt))
                except Exception:
                    text.append("NA")

        crop_image, line_coor = zip(*crops)
        Result_Df = makeDfwithfactors(Name_of_Images, Conv_factor, Scale_Mode, text, line_coor)

    else:
        Result_Df = makeDfwithfactors(Name_of_Images, Conv_factor, Scale_Mode)
    return Result_Df


### Error we put "NA" for nondetects None does not work. However NA gets split and cant be calculated in SCalemeasure which then fails




