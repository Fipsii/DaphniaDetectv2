import logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import easyocr
import cv2
import matplotlib.pyplot as plt
import re
import pandas as pd

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

def find_contrasting_horizontal_line_in_monochrome_box(box_crop, visualize_randomized=False, tolerance=5):
    gray = safe_to_gray(box_crop)
    
    # Find most common pixel value (background)
    vals, counts = np.unique(gray, return_counts=True)

    background_val = vals[np.argmax(counts)]

    # Randomize background pixels
    gray_mod = gray.copy()
    background_mask = (gray == background_val)
    num_bg_pixels = np.sum(background_mask)
    gray_mod[background_mask] = np.random.randint(0, 256, size=num_bg_pixels)

    if visualize_randomized:
        plt.figure(figsize=(6,4))
        plt.title("Randomized Background Pixels in Grayscale Box")
        plt.imshow(gray_mod, cmap='gray')
        plt.axis('off')
        plt.show()

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



import numpy as np
import cv2
import numpy as np

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



# --- 1. Helper Function: The Walker (from your original code) ---
def find_edge(img, start, step, color_threshold=30):
    h, w = img.shape[:2]
    x, y = start
    
    # helper for checking blackness
    def is_pixel_black(p):
        if isinstance(p, (int, float, np.uint8)):
            return p < color_threshold
        return np.all(p < color_threshold)

    while 0 <= x + step[0] < w and 0 <= y + step[1] < h:
        next_pixel = img[y + step[1], x + step[0]]
        if not is_pixel_black(next_pixel):
            break
        x += step[0]
        y += step[1]
    return [x, y]

# --- 2. Main Function: Your Optimized scan_for_box ---
def scan_for_box(img, color_threshold=30, min_box_size=50, max_box_percent=0.25):
    """
    Scans for a black box with added checks to reject large or central objects.
    max_box_percent: Max allowed area of the box relative to the full image (e.g., 0.25 = 25%).
    """
    h, w = img.shape[:2]
    img_area = w * h

    # 1. FAST LOOKUP & SORT (Same as before)
    if img.ndim == 3:
        is_black_mask = np.all(img < color_threshold, axis=-1)
    else:
        is_black_mask = img < color_threshold
    ys, xs = np.nonzero(is_black_mask)
    if len(xs) == 0: return [None, None]
    sort_order = np.lexsort((ys, -xs))

    # 2. CHECK CANDIDATES
    for i in sort_order:
        px, py = xs[i], ys[i]
        if px % 5 != 0: continue

        # --- Run the walker ---
        top_left = find_edge(img, [px, py], [-1, 0], color_threshold)
        top_right = find_edge(img, [px, py], [1, 0], color_threshold)
        bottom_right = find_edge(img, top_right, [0, 1], color_threshold)

        width = abs(top_right[0] - top_left[0])
        height = abs(bottom_right[1] - top_right[1])
        box_area = width * height

        # --- NEW VALIDATION CHECKS ---

        # A. Size Check: Must be within min and max limits
        if width < min_box_size or height < min_box_size:
            continue
        if box_area > (img_area * max_box_percent):
            # Reject boxes that are too huge (like the dark circle)
            continue

        # B. Location Check: Reject boxes near the center
        box_center_x = top_left[0] + width // 2
        box_center_y = top_left[1] + height // 2
        
        # Define the central 50% of the image
        center_x_min, center_x_max = w * 0.25, w * 0.75
        center_y_min, center_y_max = h * 0.25, h * 0.75
        
        if (center_x_min < box_center_x < center_x_max) and \
           (center_y_min < box_center_y < center_y_max):
             # Box is in the center, likely not a scale bar
             continue
             
        # If all checks pass, this is a good box. Return it.
        return [top_left, bottom_right]

    return [None, None]

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
    top_left, bottom_right = scan_for_box(image)

    if top_left is not None and bottom_right is not None and not all(abs(a - b) <= 20 for a, b in zip(top_left, bottom_right)):
        #print("Scale box detected. Finding scale")
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


def Detect(box_crop, reader_instance):
    results = []
    for item in box_crop:
        if item[0] is not None:
            crop_reworked = scale_for_easyocr(item[0])
            # Use the global reader_instance
            result = reader_instance.readtext(crop_reworked, allowlist='0123456789.µuUmMnNIi')
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


def improve_crop_for_ocr(crop):
    """
    1. Binarizes (Black/White only)
    2. Inverts colors if needed (ensures Black text on White bg)
    3. Upsamples (3x) and Sharpens
    """
    # 1. Ensure Grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    binary = gray.copy()

    # 3. Fix Inversion (We want Black Text on White Background)
    # Count white pixels. If white is the minority (text), it means we have 
    # White Text on Black Background -> Invert it.
    num_white = cv2.countNonZero(binary)
    if num_white < binary.size / 2:
        binary = cv2.bitwise_not(binary)

    # 4. Upsample (3x) and Sharpen
    scale_factor = 3
    upsampled = cv2.resize(binary, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
    
    # Sharpening Kernel
    kernel = np.array([[0, -1, 0], 
                       [-1, 5, -1], 
                       [0, -1, 0]])
    sharpened = cv2.filter2D(upsampled, -1, kernel)
    
    return sharpened

# --- MAIN FUNCTION ---
def process_folder(List_of_paths):
    all_crops = []
    all_results = []
    
    # Initialize Reader ONCE here to save time
    reader = easyocr.Reader(['en'])

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
                # --- APPLY IMPROVEMENT HERE ---
                # This replaces the old equalize/threshold logic
                processed_img = improve_crop_for_ocr(crop[0])
                
                crop_data = (processed_img, crop[1])
                all_crops.append(crop_data)
                all_results.append(None)  # Placeholder
            except Exception as e:
                print(f"[ERROR] Processing failed for {path}: {e}")
                all_crops.append((None, None))
                all_results.append("NA")
        else:
            all_crops.append((None, None))
            all_results.append("NA")

    # Run detection on all valid crops
    valid_crops = [c for c in all_crops if c[0] is not None and c[1] is not None]

    # --- VISUALIZE SHARPENED CROPS ---
    #for i, crop_data in enumerate(valid_crops):
        #image_pixels = crop_data[0]
        
        #plt.figure(figsize=(4, 2))
        #plt.imshow(image_pixels, cmap='gray')
        #plt.title(f"Valid Crop #{i+1} (Sharpened)")
        #plt.axis('off')
        #plt.show()
    # ---------------------------------

    if valid_crops:
        # Pass the global reader instance
        # Ensure your Detect function does NOT re-scale the image, 
        # as we already upsampled it in improve_crop_for_ocr!
        valid_results = Detect(valid_crops, reader)

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
    
    print(List_of_scale_numbers)
    LengthOpt = int(max(set([x for x in list_of_lengths if x != "NA"]), key=list_of_lengths.count))
    # Keep only numbers (int or float)
    valid_numbers = [x for x in List_of_scale_numbers if isinstance(x, (int, float))]

    if valid_numbers:
        UnitOpt = float(max(set(valid_numbers), key=valid_numbers.count))
    else:
        UnitOpt = None  # or some default value


  # Different Scales
  elif Scale_Mode == 2:

    try:
    	LengthOpt = [int(x) for x in list_of_lengths]
    	UnitOpt = [float(x) for x in List_of_scale_numbers]
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

    if Scale_Mode != 0:
        results, crops = process_folder(Paths_of_Images)

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




