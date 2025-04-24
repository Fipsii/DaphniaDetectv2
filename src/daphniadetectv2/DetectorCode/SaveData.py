import pandas as pd
import os
from PIL import Image

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
