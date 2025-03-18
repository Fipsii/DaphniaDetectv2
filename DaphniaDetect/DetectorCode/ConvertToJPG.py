##### Convert all non jpg to jpg, Ignores files that cant be made to jpg. 

def ConvertToJPEG(directory, save_loc):
  import os as os
  from PIL import Image
  import shutil
  # Input: directory of Image folder
  # Savefolder, in which the JPG folder is created
  # Save the folder adjacent to the real images and call it JPG
  parent_dir = os.path.dirname(save_loc)
  save_folder = os.path.join(parent_dir, "JPG")

  if os.path.exists(save_folder):
    shutil.rmtree(save_folder)
  
  os.mkdir(save_folder)

  for root, dirs, files in os.walk(directory, topdown=False):

    counter = 0
    for name in files:
      counter +=1
      outfile = os.path.splitext(os.path.join(root, name))[0] + ".jpg"
      Image_name = os.path.basename(outfile)
      try:
        im = Image.open(os.path.join(root, name))
        im.thumbnail(im.size)
        im.save(os.path.join(save_folder, Image_name), "JPEG", quality=100)
        
        ## Print progress
        progress_bar(counter, len(files))
      
      except Exception:
        print(f"Could not save {os.path.splitext(os.path.join(root, name))}")

  print() ## Makes a new line
  return(save_folder)


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
