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

def ConvertToJPEG(directory, save_loc):
    """
    Converts all images in 'directory' to JPEG and moves existing JPEGs to a 'JPG' folder.
    
    Args:
    - directory (str): Path to the folder containing images.
    - save_loc (str): Path where the 'JPG' folder will be created.

    Returns:
    - save_folder (str): Path of the newly created JPG folder.
    """

    # Define the save folder
    parent_dir = os.path.dirname(save_loc)
    save_folder = os.path.join(parent_dir, "JPG")

    # Delete the old JPG folder if it exists
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)

    os.mkdir(save_folder)

    counter = 0  # Progress counter

    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            outfile = os.path.splitext(name)[0] + ".jpg"
            save_path = os.path.join(save_folder, outfile)

            try:
                # Open the image
                im = Image.open(file_path)

                # If the image has an alpha channel (RGBA), remove transparency
                if im.mode == "RGBA":
                    background = Image.new("RGB", im.size, (255, 255, 255))  # White background
                    background.paste(im, mask=im.split()[3])  # Apply transparency mask
                    im = background
                else:
                    im = im.convert("RGB")  # Convert to RGB if not already

                # Save as JPEG
                im.save(save_path, "JPEG", quality=100)

                counter += 1  # Update counter
                print(f"Converted: {name} -> {outfile}")

            except Exception as e:
                print(f"Could not convert {name}: {e}")

    print(f"\nConversion complete! Total images processed: {counter}")
    return save_folder



def progress_bar(iterations, total):
    ## Display a progress bar for time intesive steps
    ## Input: Task iterations, total iterations
    ## Output: Progress bar in Terminal
    length = 50  # Length of the progress bar in characters
    progress = int(length * iterations / total)
    block = "█"
    space = " "
    bar = block * progress + space * (length - progress)
    percentage = 100 * iterations / total
    print(f"\rGenerating jpgs |{bar}| {percentage:.1f}% Complete", end="", flush = True)
    

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
