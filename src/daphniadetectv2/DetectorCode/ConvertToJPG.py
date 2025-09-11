##### Convert all non jpg to jpg, Ignores files that cant be made to jpg. 

import os

def CheckJPG(image_dir):
    """Check if all images in a directory are already JPEGs."""
    valid_extensions = {".jpg", ".jpeg"}
    for filename in os.listdir(image_dir):
        if os.path.isfile(os.path.join(image_dir, filename)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in valid_extensions:
                return False  # Found a non-JPEG file
    return True  # All images are JPEGs
  
import os
import shutil
from PIL import Image

import os
import shutil
from PIL import Image, ImageEnhance
import numpy as np
import cv2  # Only for CLAHE, if we keep OpenCV just for that

def apply_clahe_with_pil(pil_img):
    """
    Apply CLAHE-like enhancement to a PIL image using OpenCV for LAB processing.
    """
    # Convert PIL to OpenCV (BGR)
    img_cv = np.array(pil_img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # Convert to LAB and apply CLAHE on L channel
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge and convert back to BGR
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Convert back to PIL (RGB)
    enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(enhanced_rgb)


import os
import shutil
from PIL import Image
import cv2
import numpy as np

def apply_clahe_with_pil(pil_img):
    """
    Converts a PIL image to grayscale, applies CLAHE using OpenCV, and returns a PIL image.
    """
    # Convert PIL to numpy array
    img_np = np.array(pil_img.convert("L"))  # Convert to grayscale

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img_np)

    # Convert back to PIL
    return Image.fromarray(cl_img)

import os
import shutil
from PIL import Image

def ConvertToJPEG(directory, save_loc):
    """
    Converts all images in 'directory' to grayscale JPEG with CLAHE enhancement,
    and moves existing JPEGs to a 'JPG' folder.

    Args:
    - directory (str): Path to the folder containing images.
    - save_loc (str): Path where the 'JPG' folder will be created.

    Returns:
    - save_folder (str): Path of the newly created JPG folder.
    """

    parent_dir = os.path.dirname(save_loc)
    save_folder = os.path.join(parent_dir, "JPG")

    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.mkdir(save_folder)

    # Count total images to convert
    all_files = [
        os.path.join(root, name)
        for root, _, files in os.walk(directory, topdown=False)
        for name in files
    ]
    total_files = len(all_files)
    counter = 0

    for idx, file_path in enumerate(all_files):
        name = os.path.basename(file_path)
        outfile = os.path.splitext(name)[0] + ".jpg"
        save_path = os.path.join(save_folder, outfile)

        try:
            im = Image.open(file_path)

            if im.mode == "RGBA":
                background = Image.new("RGB", im.size, (255, 255, 255))
                background.paste(im, mask=im.split()[3])
                im = background
            else:
                im = im.convert("RGB")

            # Save as grayscale JPEG (or just RGB for now)
            im.save(save_path, "JPEG", quality=100)

            counter += 1
        except Exception as e:
            print(f"\nCould not convert {name}")

        # Update progress bar
        progress_bar(idx + 1, total_files)
    return save_folder


def progress_bar(iterations, total):
    length = 50  # Length of the progress bar
    progress = int(length * iterations / total)
    bar = '█' * progress + ' ' * (length - progress)
    percentage = 100 * iterations / total
    print(f"\rGenerating JPGs |{bar}| {percentage:.1f}% Complete", end="", flush=True)


def EmptyCheck(folder_path):
    ###Check if a folder is empty.	
    ###Args:
    #   folder_path (str): The path to the folder to check.
    #Returns:
    #    bool: True if the folder is empty, False otherwise.
    
    import os

    try:
        # List the contents of the folder
        folder_contents = os.listdir(folder_path)
        if len(folder_contents) == 0:
          print("No images in folder. Check your input folder.") 
          exit()
        else:
           pass
    except FileNotFoundError:
        print(f"The folder '{folder_path}' does not exist.")
        exit()
