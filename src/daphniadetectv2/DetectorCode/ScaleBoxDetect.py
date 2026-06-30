import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def find_scale_line_near_text(image_strip, text_y_min, text_y_max, text_x_min, text_x_max):
    """
    Detects the scale bar by structurally isolating the enclosing box 
    and extracting the longest continuous horizontal line.
    """
    gray = cv2.cvtColor(image_strip, cv2.COLOR_BGR2GRAY) if len(image_strip.shape) == 3 else image_strip
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 1. Isolate the bounding contour mathematically
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    box_rect = None
    min_area = float('inf')
    img_area = gray.shape[0] * gray.shape[1]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Verify contour encapsulates the text coordinates
        if (x <= text_x_min + 5 and y <= text_y_min + 5 and 
            x + w >= text_x_max - 5 and y + h >= text_y_max - 5):
            
            area = w * h
            if area < min_area and area < img_area * 0.95:
                min_area = area
                box_rect = (x, y, w, h)
                
    # 2. Extract strictly bounded ROI
    if box_rect:
        bx, by, bw, bh = box_rect
        m = 3 # Margin to reject perimeter pixels
        x1, y1 = max(0, bx + m), max(0, by + m)
        x2, y2 = min(gray.shape[1], bx + bw - m), min(gray.shape[0], by + bh - m)
    else:
        # Fallback heuristic
        pad_y = int(max(10, (text_y_max - text_y_min) * 1.5))
        pad_x = int(max(10, (text_x_max - text_x_min) * 0.5))
        x1, y1 = max(0, int(text_x_min) - pad_x), max(0, int(text_y_min) - pad_y)
        x2, y2 = min(gray.shape[1], int(text_x_max) + pad_x), min(gray.shape[0], int(text_y_max) + pad_y)

    if x2 <= x1 or y2 <= y1: return None
    
    roi_binary = binary[y1:y2, x1:x2].copy()
    
    # 3. Morphological vector filtering
    # Eliminate vertical components
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 10))
    v_open = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_v)
    roi_binary[v_open > 0] = 0
    
    # Eliminate short horizontal/diagonal components (text residue)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    cleaned = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel_h)
    
    # 4. Extract dominant horizontal line segment
    best_row = -1
    max_len = 0
    best_segment = (0, 0)
    
    for y in range(cleaned.shape[0]):
        row = cleaned[y, :]
        diff = np.diff(np.concatenate(([0], row > 0, [0])).astype(int))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]
        
        for s, e in zip(starts, ends):
            length = e - s
            if length > (x2 - x1) * 0.2:
                if length > max_len:
                    max_len = length
                    best_row = y
                    best_segment = (s, e)
                    
    if best_row == -1: return None
    
    return (
        int(x1 + best_segment[0]),
        int(y1 + best_row),
        int(x1 + best_segment[1]),
        int(y1 + best_row)
    )

def process_folder(directory_path):
    """
    Iterates over image files in a directory, applies detection, 
    and visualizes the results.
    """
    valid_extensions = ("*.png", "*.png", "*.png", "*.tif", "*.tiff")
    files = []
    for ext in valid_extensions:
        files.extend(glob.glob(os.path.join(directory_path, ext)))
        files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
    for file_path in files:
        img = cv2.imread(file_path)
        if img is None:
            continue
            
        # IMPORTANT: Replace these mock coordinates with the actual 
        # text bounding box coordinates derived from your OCR pipeline.
        h, w = img.shape[:2]
        text_x_min, text_x_max = int(w * 0.3), int(w * 0.7)
        text_y_min, text_y_max = int(h * 0.6), int(h * 0.9)
        
        coords = find_scale_line_near_text(img, text_y_min, text_y_max, text_x_min, text_x_max)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Render the input text bounding box boundary (blue dashed line)
        rect = plt.Rectangle((text_x_min, text_y_min), text_x_max - text_x_min, text_y_max - text_y_min, 
                             linewidth=1, edgecolor='blue', facecolor='none', linestyle='--')
        ax.add_patch(rect)
        
        # Render the detected scale bar line (red solid line)
        if coords:
            x1, y1, x2, y2 = coords
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=3)
        
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == "__main__":
    # Specify the directory containing your scale bar crops
    folder_path = r"C:\Users\hanss\Desktop\Daphnia\2025_Images_Sorted\Test_Images\No_Measure\Marius"
    process_folder(folder_path)