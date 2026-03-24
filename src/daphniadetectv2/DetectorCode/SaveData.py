import pandas as pd
import os
from PIL import Image

class_names = {
    0: "Body",
    1: "Brood cavity",
    2: "Daphnia",
    3: "Eye",
    4: "Head",
    5: "Heart",
    6: "Spina base",
    7: "Spina tip",
    8: "SpinaTipBase"
}

def read_yolo_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    bboxes = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = parts
        class_id = int(float(class_id))
        organ_name = class_names.get(class_id, f'organ{class_id}')
        
        # Store coordinates in a nested dictionary instead of flat columns
        bboxes[organ_name] = {
            'x_center': float(x_center),
            'y_center': float(y_center),
            'width': float(width),
            'height': float(height)
        }
    return {'bboxes': bboxes}
    
      
def read_yolo_folder(folder_path, image_folder):
    all_data = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            file_data = read_yolo_file(file_path)

            image_name = filename.replace('.txt', '.jpg')
            image_path = os.path.join(image_folder, image_name)

            # Try to open image and get its dimensions
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                    file_data['image_width'] = width
                    file_data['image_height'] = height
            except FileNotFoundError:
                file_data['image_width'] = None
                file_data['image_height'] = None

            # Crucial: Use 'image_name' to match downstream functions
            file_data['image_name'] = image_name
            all_data.append(file_data)

    return pd.DataFrame(all_data)


def convert_yolo_to_pixel(df):
    df_pixel = df.copy()

    for idx, row in df_pixel.iterrows():
        # Skip if bboxes aren't present or aren't a dict
        if 'bboxes' not in row or not isinstance(row['bboxes'], dict):
            continue
            
        img_w = row.get('image_width')
        img_h = row.get('image_height')
        
        if pd.isna(img_w) or pd.isna(img_h):
            continue
            
        updated_bboxes = {}
        for organ, coords in row['bboxes'].items():
            new_coords = coords.copy()
            
            # Calculate pixel centers and dimensions
            new_coords['pixel_x_center'] = coords['x_center'] * img_w
            new_coords['pixel_y_center'] = coords['y_center'] * img_h
            new_coords['pixel_width'] = coords['width'] * img_w
            new_coords['pixel_height'] = coords['height'] * img_h
            
            # Calculate min and max bounds
            new_coords['x_min'] = int((coords['x_center'] - coords['width'] / 2) * img_w)
            new_coords['y_min'] = int((coords['y_center'] - coords['height'] / 2) * img_h)
            new_coords['x_max'] = int((coords['x_center'] + coords['width'] / 2) * img_w)
            new_coords['y_max'] = int((coords['y_center'] + coords['height'] / 2) * img_h)
            
            updated_bboxes[organ] = new_coords
            
        df_pixel.at[idx, 'bboxes'] = updated_bboxes

    return df_pixel
    
import pandas as pd
import os
from PIL import Image

### Save Data if only Segmentation was performed ######
# Define a mapping from class ID to organ name or number
class_names = {
                    0: "Body",
                    1: "Brood cavity",
                    2: "Daphnia",
                    3: "Eye",
                    4: "Head",
                    5: "Heart",
                    6: "Spina base",
                    7: "Spina tip",
                    8: "SpinaTipBase"
                }
  

def merge_confidence(df_meas, df_conf, key_col='image_name'):
    """
    Merges confidence scores from df_conf into the dictionary columns of df_meas.
    """
    # 1. Create copies to avoid modifying the original dataframes
    meas = df_meas.copy()
    conf = df_conf.copy()
    
    # 2. Set the key column as index for proper alignment
    meas = meas.set_index(key_col, drop=False)
    conf = conf.set_index(key_col, drop=False)
    
    # 3. Align df_conf to match the exact row order and index of df_meas
    # Note: df_conf must have unique image_names for this to work safely.
    conf = conf.reindex(meas.index)

    # 4. Identify columns to process (intersection of columns, excluding the key)
    organ_cols = [c for c in meas.columns if c in conf.columns and c != key_col]

    # 5. Helper function to update a single cell
    def update_single_cell(box_data, conf_value):
        # A. Check if confidence is missing (NaN)
        if pd.isna(conf_value):
            return box_data
            
        # B. Check if box_data is valid. 
        # If it is None or a float (NaN), we cannot update it.
        if box_data is None or (isinstance(box_data, float) and pd.isna(box_data)):
            return box_data

        # C. Parse data if it is a string representation of a dict
        if isinstance(box_data, str):
            try:
                d = ast.literal_eval(box_data)
            except (ValueError, SyntaxError):
                return box_data 
        elif isinstance(box_data, dict):
            d = box_data.copy()
        else:
            return box_data

        # D. Insert the confidence value
        d['conf'] = conf_value
        return d

    # --- THIS WAS MISSING IN YOUR SNIPPET ---
    # 6. Apply the logic to each shared column
    for col in organ_cols:
        meas[col] = [
            update_single_cell(m, c) 
            for m, c in zip(meas[col], conf[col])
        ]

    # 7. Return the result
    return meas.reset_index(drop=True)
    
    
    
    
    
import pandas as pd
import ast
import re

def flatten_and_merge_data(Measurements, Confidence):
    def clean_and_parse_bboxes(bbox_str):
        if pd.isna(bbox_str): 
            return {}
        cleaned = re.sub(r'np\.(int|float)64\((.*?)\)', r'\2', str(bbox_str))
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError):
            return {}

    def process_row(row):
        bboxes_dict = clean_and_parse_bboxes(row.get('bboxes', None))
        result = {}
        for organ, detection in bboxes_dict.items():
            
            if isinstance(detection, list) and detection:
                result[organ] = detection[0] 
            elif isinstance(detection, dict):
                result[organ] = detection
        return pd.Series(result)

    # 1. Expand bboxes into organ columns
    organ_cols = Measurements.apply(process_row, axis=1)

    # 2. Retain ALL original columns except the unparsed 'bboxes'
    df_meta = Measurements.drop(columns=['bboxes']) if 'bboxes' in Measurements.columns else Measurements.copy()
    df_condensed = pd.concat([df_meta, organ_cols], axis=1)

    # 3. Pivot Confidence Data
    conf_renames = {'Image': 'image_name', 'Class': 'organ', 'class': 'organ', 'Confidence': 'confidence', 'conf': 'confidence'}
    df_conf = Confidence.rename(columns=conf_renames)
    
    df_conf_pivoted = df_conf.pivot(index='image_name', columns='organ', values='confidence')
    df_conf_pivoted.columns = [f"{col}_confidence" for col in df_conf_pivoted.columns]
    df_conf_pivoted = df_conf_pivoted.reset_index()

    # 4. Merge
    final_df = pd.merge(df_condensed, df_conf_pivoted, on='image_name', how='left')

    # 5. Sort columns to keep organs and their confidence scores together
    sorted_cols = list(df_meta.columns)
    found_organs = list(organ_cols.columns)

    for organ in found_organs:
        if organ in final_df.columns:
            sorted_cols.append(organ)
        conf_col = f"{organ}_confidence"
        if conf_col in final_df.columns:
            sorted_cols.append(conf_col)

    remaining_cols = [c for c in final_df.columns if c not in sorted_cols]
    return final_df[sorted_cols + remaining_cols]


    
def export_to_nested_json(final_df, output_path):
    # Set the target key as the DataFrame index
    # We also delete the mask it saved in folder
    if 'mask' in final_df.columns:
        final_df = final_df.drop(columns=['mask', 'rotated_mask'])

    # Set the target key as the DataFrame index
    if 'image_name' in final_df.columns:
        indexed_df = final_df.set_index('image_name')
    else:
        indexed_df = final_df
        
    indexed_df.to_json(output_path, orient='index', indent=4)