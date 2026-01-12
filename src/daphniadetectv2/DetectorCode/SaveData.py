import pandas as pd
import os
from PIL import Image
import re
import ast
### Save Data if only Segmentation was performed ######
# Define a mapping from class ID to organ name or number
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
  

def read_yolo_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    row = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        class_id, x_center, y_center, width, height = parts
        class_id = int(float(class_id))
        organ_name = class_names.get(class_id, f'organ{class_id}')
        row.update({
            f'x_center_{organ_name}': float(x_center),
            f'y_center_{organ_name}': float(y_center),
            f'width_{organ_name}': float(width),
            f'height_{organ_name}': float(height)
        })
    return row
    
      
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

            file_data['filename'] = image_name
            all_data.append(file_data)

    df = pd.DataFrame(all_data)

    # Ensure all columns exist
    all_organs = set(class_names.values())
    full_columns = []
    for organ in all_organs:
        full_columns += [
            f'x_center_{organ}', f'y_center_{organ}',
            f'width_{organ}', f'height_{organ}'
        ]
    full_columns += ['filename', 'image_width', 'image_height']
    for col in full_columns:
        if col not in df.columns:
            df[col] = None
    df = df[sorted(df.columns)]

    return df


def convert_yolo_to_pixel(df):
    df_pixel = df.copy()

    for organ in class_names.values():
        x_col = f'x_center_{organ}'
        y_col = f'y_center_{organ}'
        w_col = f'width_{organ}'
        h_col = f'height_{organ}'

        if x_col in df_pixel.columns:
            df_pixel[f'pixel_x_center_{organ}'] = df_pixel[x_col] * df_pixel['image_width']
        if y_col in df_pixel.columns:
            df_pixel[f'pixel_y_center_{organ}'] = df_pixel[y_col] * df_pixel['image_height']
        if w_col in df_pixel.columns:
            df_pixel[f'pixel_width_{organ}'] = df_pixel[w_col] * df_pixel['image_width']
        if h_col in df_pixel.columns:
            df_pixel[f'pixel_height_{organ}'] = df_pixel[h_col] * df_pixel['image_height']

    return df_pixel


def clean_and_parse_bboxes(bbox_str):
    """
    Cleans numpy type indicators from the string and parses it into a dict.
    """
    if pd.isna(bbox_str):
        return {}
    
    # Remove 'np.int64(' and ')' 
    # We use regex to be safe: matches "np.int64(" or "np.float64("
    cleaned = re.sub(r'np\.(int|float)64\((.*?)\)', r'\2', str(bbox_str))
    
    try:
        return ast.literal_eval(cleaned)
    except (ValueError, SyntaxError):
        return {}

def process_row(row):
    """
    Extracts organ data from the 'bboxes' column for a single row.
    """
    bboxes_dict = clean_and_parse_bboxes(row['bboxes'])
    result = {}
    
    # Iterate through keys like 'Body', 'Head', 'Eye', etc.
    for organ, detection_list in bboxes_dict.items():
        if detection_list:
            # Take the first detection (usually the most confident/primary one)
            result[organ] = detection_list[0]
            
    return pd.Series(result)


def format_df(df):
    """
    Formats the data.csv df into a more readable format 
    """
    # 2. Extract organ data into new columns
    organ_data = df.apply(process_row, axis=1)

    # 3. Select only the lightweight metadata columns
    metadata_cols = ['image_name', 'metric_length', 'scale[px]', 'distance_per_pixel', 'species']
    # Use intersection to avoid errors if a column is missing
    existing_meta_cols = [c for c in metadata_cols if c in df.columns]
    metadata = df[existing_meta_cols]

    # 4. Concatenate and Print
    condensed_df = pd.concat([metadata, organ_data], axis=1)

    return condensed_df

import pandas as pd
import ast

def merge_confidence(df_meas, df_conf, key_col='image_name'):
    """
    Merges confidence scores from df_conf into the dictionary columns of df_meas.
    
    Args:
        df_meas (pd.DataFrame): DataFrame containing measurement dicts (or strings of dicts).
        df_conf (pd.DataFrame): DataFrame containing confidence floats.
        key_col (str): The common column name to join on (e.g., 'image_name').
        
    Returns:
        pd.DataFrame: A new DataFrame with 'conf' added to the measurement dictionaries.
    """
    # 1. Create copies to avoid modifying the original dataframes
    meas = df_meas.copy()
    conf = df_conf.copy()
    
    # 2. Set the key column as index for proper alignment
    # This ensures we match the correct row even if the sort order differs
    meas = meas.set_index(key_col, drop=False)
    conf = conf.set_index(key_col, drop=False)
    
    # 3. Align df_conf to match the exact row order and index of df_meas
    # This discards rows in conf that aren't in meas, and adds NaNs if meas has rows conf doesn't
    conf = conf.reindex(meas.index)

    # 4. Identify columns to process (intersection of columns, excluding the key)
    organ_cols = [c for c in meas.columns if c in conf.columns and c != key_col]

    # 5. Helper function to update a single cell
    def update_single_cell(box_data, conf_value):
        # If either data point is missing, return the original box data unchanged
        if pd.isna(conf_value) or box_data is None or pd.isna(box_data):
            return box_data

        # If data is a string representation of a dict (common in CSVs), parse it
        if isinstance(box_data, str):
            try:
                d = ast.literal_eval(box_data)
            except (ValueError, SyntaxError):
                return box_data 
        elif isinstance(box_data, dict):
            d = box_data.copy()
        else:
            return box_data

        # Insert the confidence value
        d['conf'] = conf_value
        return d

    # 6. Apply the logic to each shared column
    for col in organ_cols:
        # We use zip because we have already forced the indices to align perfectly in step 3
        meas[col] = [
            update_single_cell(m, c) 
            for m, c in zip(meas[col], conf[col])
        ]

    # 7. Clean up and return
    return meas.reset_index(drop=True)