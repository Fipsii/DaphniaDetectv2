from ultralytics import YOLO
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from pathlib import Path

ID_BROOD, ID_EYE, ID_HEART, ID_SPINA = 1, 3, 5, 6

def parse_yolo(txt_path, img_w, img_h):
    coords, boxes = {}, {}
    if not txt_path.exists(): return coords, boxes
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5: continue
            c_id, x_c, y_c, w, h = map(float, parts)
            cid = int(c_id)
            coords[cid] = np.array([x_c * img_w, y_c * img_h])
            x_min, y_min = int((x_c - w/2)*img_w), int((y_c - h/2)*img_h)
            x_max, y_max = int((x_c + w/2)*img_w), int((y_c + h/2)*img_h)
            boxes[cid] = [max(0, x_min), max(0, y_min), min(img_w, x_max), min(img_h, y_max)]
    return coords, boxes

def extract_ventral_notch(mask_array, sigma=3.0):
    h, w = mask_array.shape
    mask_uint8 = (mask_array * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: 
        return None
        
    c = max(contours, key=cv2.contourArea).squeeze()
    if c.ndim != 2 or len(c) < 10: 
        return None

    x_smooth = gaussian_filter1d(c[:, 0].astype(float), sigma=sigma, mode='wrap')
    y_smooth = gaussian_filter1d(c[:, 1].astype(float), sigma=sigma, mode='wrap')
    smoothed_contour = np.column_stack((x_smooth, y_smooth))

    cx, cy = np.mean(x_smooth), np.mean(y_smooth)

    n = len(c)
    valid_mask = np.zeros(n, dtype=bool)
    margin_y_top = int(h * 0.30)
    margin_y_bottom = int(h * 0.30)
    margin_x = int(w * 0.05)
    
    for i in range(n):
        x, y = smoothed_contour[i]
        if x <= margin_x or x >= w - margin_x: continue
        if y <= margin_y_top or y >= h - margin_y_bottom: continue
        if x < cx: continue 
        valid_mask[i] = True
        
    if not valid_mask.any():
        return None
        
    valid_indices = np.where(valid_mask)[0]
    valid_pts = smoothed_contour[valid_indices]
    
    centroid = np.array([cx, cy])
    distances = np.linalg.norm(valid_pts - centroid, axis=1)
    inflection_idx = valid_indices[np.argmin(distances)]
    refined_pt = tuple(smoothed_contour[inflection_idx])
    
    return refined_pt

def Refine_spine_base(img_dir, label_dir, seg_model_path):
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    
    seg_model = YOLO(seg_model_path)
    img_files = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    
    for img_path in img_files:
        yolo_path = label_dir / f"{img_path.stem}.txt"
        if not yolo_path.exists(): continue

        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        
        # Read current lines into memory to prevent I/O conflict during overwrite
        with open(yolo_path, 'r') as f:
            original_lines = f.readlines()
        
        coords, boxes = parse_yolo(yolo_path, w, h)
        
        if ID_EYE not in coords or ID_SPINA not in coords or ID_SPINA not in boxes: continue
        
        ref_coord = coords.get(ID_BROOD, coords.get(ID_HEART))
        if ref_coord is None: continue

        # Coordinate Mapping Geometry
        E, S, P = coords[ID_EYE], coords[ID_SPINA], ref_coord
        
        delta = E - S
        angle_deg = np.degrees(np.arctan2(delta[1], delta[0]))
        rot_mat = cv2.getRotationMatrix2D(tuple(S), angle_deg + 90, 1.0)
        rotated_img = cv2.warpAffine(img, rot_mat, (w, h), flags=cv2.INTER_CUBIC)

        P_rot = rot_mat @ np.array([P[0], P[1], 1.0])
        S_rot = rot_mat @ np.array([S[0], S[1], 1.0])

        is_flipped = False
        if P_rot[0] > S_rot[0]:
            rotated_img = cv2.flip(rotated_img, 1)
            S_rot[0] = w - S_rot[0] 
            is_flipped = True

        b = boxes[ID_SPINA]
        side = max(b[2] - b[0], b[3] - b[1])  
        half = int(side / 2)
        x_c, y_c = int(S_rot[0]), int(S_rot[1])
        y1, y2 = max(0, y_c - half), min(h, y_c + half)
        x1, x2 = max(0, x_c - half), min(w, x_c + half)
        
        crop = rotated_img[y1:y2, x1:x2].copy()
        if crop.size == 0: continue

        ch, cw = crop.shape[:2]
        border_w = max(1, int(cw * 0.10))
        crop[:, cw - border_w:] = [255, 255, 255]

        # Segmentation & Calculation
        results = seg_model.predict(source=crop, imgsz=256, conf=0.5, verbose=False)
        if not results or results[0].masks is None: continue
            
        mask_data = results[0].masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask_data, (cw, ch), interpolation=cv2.INTER_NEAREST)
        
        refined_pt = extract_ventral_notch(mask_resized, sigma=3.0)
        
        if refined_pt is not None:
            # Inverse Coordinate Mapping
            rx_rot_flip = refined_pt[0] + x1
            ry_rot_flip = refined_pt[1] + y1

            if is_flipped:
                rx_rot = w - rx_rot_flip
            else:
                rx_rot = rx_rot_flip
            ry_rot = ry_rot_flip

            inv_rot_mat = cv2.invertAffineTransform(rot_mat)
            orig_pt = inv_rot_mat @ np.array([rx_rot, ry_rot, 1.0])
            new_spina_x = orig_pt[0] / w
            new_spina_y = orig_pt[1] / h

            # Overwrite original annotations
            with open(yolo_path, 'w') as fw:
                for line in original_lines:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        fw.write(line)
                        continue
                        
                    c_id = int(float(parts[0]))
                    if c_id == ID_SPINA:
                        w_norm, h_norm = float(parts[3]), float(parts[4])
                        fw.write(f"{c_id} {new_spina_x:.6f} {new_spina_y:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    else:
                        fw.write(line)


