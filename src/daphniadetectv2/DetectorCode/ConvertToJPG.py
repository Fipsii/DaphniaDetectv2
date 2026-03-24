import os
import shutil
import cv2
import numpy as np
from PIL import Image


def convert_to_jpeg(source_dir, output_parent):
    """
    Converts images in source_dir to grayscale JPEGs with CLAHE.
    Skips files that cannot be processed.
    """
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return

    save_folder = os.path.join(output_parent, "JPG")
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        save_path = os.path.join(save_folder, os.path.splitext(filename)[0] + ".jpg")

        try:
            with Image.open(file_path) as im:
                # Handle transparency (RGBA/P) by flattening onto white background
                if im.mode in ("RGBA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[3])
                    im = bg
                
                # Apply CLAHE and save
                
                im.save(save_path, "JPEG", quality=95)
        except (IOError, SyntaxError, Image.DecompressionBombError):
            # Ignore non-image files or corrupted data
            continue

    print(f"Conversion complete. Files located in: {save_folder}")
    return save_folder
