### Goal: Read in an Image an autoamtically detect the Scale used:
## Caveats: Maybe we will need manual input of scale, 
##Do we perform this per Image or once provided the person doesn't change the zoom 


#### 
#### Read in all Images and save their respective paths and filenames
def Images_list(path_to_images):
  ## Takes path, creates list of image names and full paths for all
  ## PNGS or JPGS in the folder
  import os as os
  PureNames = []
  Image_names = []
  for root, dirs, files in os.walk(path_to_images, topdown=False):
    #print(dirs, files)
    for name in files:
      _, ext = os.path.splitext(name)
      if ext.lower() in ['.jpg', '.jpeg', '.png'] and name != '.DS_Store':
        #print(os.path.join(root, name))
        Image_names.append(os.path.join(root, name))
        PureNames.append(name)
        #print(files)
  return Image_names, PureNames

def getLineLength(Image_names):
  ## Gaussiaun blur and image read ##
  # Detects lines used for scale, always
  # counts the saves the shortest scale
  # in image
  # Input: image names in folder
  # Output: line coordinates and length
  ###################################
  
  import cv2 as cv2
  import numpy as np
  from PIL import Image
  import matplotlib.pyplot as plt
  
  list_of_lengths = []
  list_of_images = []
  line_Coors = []
  list_of_lines = []
  for x in range(len(Image_names)):
      img = cv2.imread(Image_names[x])

      #print(x)
      # Load the original RGB image
      normalized_image = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
      # Extract the R, G, B channels
      b, g, r = cv2.split(normalized_image)
      # Create a mask for pixels where R = G = B
      mask = np.logical_and(r == g, g == b)
      # Create a new grayscale image with the same size as the original image
      gray_image = np.zeros_like(r, dtype=np.uint8)
      # Set the pixels in the grayscale image where R = G = B to the corresponding pixel values
      gray_image[mask] = r[mask]
      
      ## why should we blur? Image lines should be sharp -> thresholding should be the most effective
      ## In thresholding we need to consider colors but should be managable if we make an OTSU and 
      ## hard cut off for white backgrounds with white scales
      kernel_size = (5,1) # We do not wnt to blur the x axis as we need the length information 
      blur_gray = cv2.GaussianBlur(gray_image,(kernel_size),0)
      
      low_threshold = 0
      high_threshold = 1
      edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
  
      #### Set HoughLinesP Parameters
      
      rho = 1 # distance resolution in pixels of the Hough grid
      theta = np.pi / 180  # angular resolution in radians of the Hough grid
      threshold = 100  # minimum number of votes (intersections in Hough grid cell)
      min_line_length = 100 # minimum number of pixels making up a line
      max_line_gap = 0  # maximum gap in pixels between connectable line segments
      line_image = np.copy(img) * 0
      line_image2 = np.copy(img) * 0  # creating a blank to draw lines on
      # Run Hough on edge detected image
      # Output "lines" is an array containing endpoints of detected line segments
      
      lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap)
 
      if lines is None: ## If no lines are found
        line_Coors.append(0)
        list_of_lengths.append(0) 
        list_of_images.append(blur_gray)
        list_of_lines.append(0)
      else: # if there are lines
      
        Coordinates, Lengths = group_lines(lines) ## combine close lines
        #print(Lengths, Coordinates)
        # Create an empty image for drawing the lines
        
        # Discard extreme lines###
        # Case 1 Long lines at image edge
        # Hough Lines sometimes shows behavior taking edges of the iamge as line
        # But y = 0 and y = image_height is never a applicable line
        
        max_y = edges.shape[0] - 1  # Subtract 1 since indexing starts from 0
        filtered_lines = []
        filtered_lengths = []
        
        for length, line in zip(Lengths, Coordinates): ## If one of the rows is the beginnig or end of the image delete
          #print(line,length)
          # and delete the length value # y is always the same for both
          if line[0][1] != max_y and line[0][1] != 0:
            
            filtered_lines.append(line)
            filtered_lengths.append(length)
        
        # Case two lines that are over the minimum 100px but over 50% smaller than
        # the other lines in the list, which would make it a fragment
        
        filtered_lines_step2= []  # List to store the filtered lines
        filtered_lengths_step2 = []
        
        max_length = max(filtered_lengths)  # Find the maximum length
        #print(max_length, filtered_lengths, x)
        if len(filtered_lengths) > 1:
            for line, length in zip(filtered_lines, filtered_lengths):
                if length < 0.5 * max_length:  # Check if length is less than 50% of max length
                    continue  # Skip this line and length
                else:
                    filtered_lines_step2.append(line)
                    filtered_lengths_step2.append(length)
        
        else: ## If we have only one value -> no list but int then
          filtered_lines_step2 = filtered_lines
          filtered_lengths_step2 = filtered_lengths
        
        ## Now we want to select for the right line We have two conditions:
        ## If we find only one or two lines we take the shortest line we find
        ## If we have more lines we take the inner lines as the scale is contained
        ## in the box. This allows resilience against not completly detected boxe edges
        ##########################################################################
        
        #print(filtered_lengths_step2,filtered_lines_step2)
        ## Get the coordinates of the shortest line
        
        if len(filtered_lines_step2) < 3: ## If only 1-2 lines left we take the 
          # shortest line we find
          Idx = filtered_lengths_step2.index(min(filtered_lengths_step2))
          Correct_Coor = filtered_lines_step2[Idx]
        
          x1, y1, x2, y2 = Correct_Coor[0]
          
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        else: 
          # If more lines are left we take a line from the middle
          # This is only robust if we 1) merged all lines correctly
          # or upper and lower box lines are detected.
          # If we detect two lines for upper boundary and none for the lower
          # as well as 1 for the real scale we select a false value
          
          Middle_line = len(filtered_lines_step2)//2
          Correct_Coor = filtered_lines_step2[Middle_line]
          x1, y1, x2, y2 = Correct_Coor[0]
          
          cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 1)

    
        ## Control Area
        test = cv2.addWeighted(img, 0.2, line_image, 0.8, 0)
        
        for line in lines: ## No plot raw lines
          for x1,y1,x2,y2 in line:
            cv2.line(line_image2,(x1,y1),(x2,y2),(255,0,0),1)
        
        test2 = cv2.addWeighted(img, 0.2, line_image2, 0.8, 0)
       
    
        
        ## Result Area
        PixelPerUnit = min(filtered_lengths_step2) ## take the min value of length as scale
  
        # Append values to lists
        line_Coors.append(Correct_Coor)
        list_of_lengths.append(PixelPerUnit) 
        list_of_images.append(img)
        list_of_lines.append(lines)
  return list_of_lengths, line_Coors, list_of_images 

def group_lines(lines):
    ### Input list of lines Coordinates generated by houghlinesP
    ### First tries to fuse lines then dimsiss short lines
    ### then selects the shortest line as the scale
    ###
    ### Output: list of "correct" lines
    
    import numpy as np
    groups = []
    fused_lines = []
    extracted_lengths = []
    # Convert lines to a list if it is a NumPy array
    if isinstance(lines, np.ndarray):
        lines = lines.tolist()
    # Sort lines based on y-coordinate
    lines.sort(key=lambda line: line[0][1])

    # Group lines that are within 3 pixels of each other
    current_group = [lines[0]]
    for line in lines[1:]:
        if line[0][1] - current_group[-1][0][3] <= 3:
            current_group.append(line)
        else:
            groups.append(current_group)
            current_group = [line]
    groups.append(current_group)

    # Fuse lines within each group
    for group in groups:
        x_min = min(line[0][0] for line in group)
        x_max = max(line[0][2] for line in group)
        y_mean = int(sum(line[0][1] for line in group) / len(group))
        fused_lines.append([[x_min, y_mean, x_max, y_mean]])
        extracted_lengths.append(abs(x_max - x_min))

    # Discard lines that are under 100 pixels in length
    filtered_lines = []
    filtered_lengths = []
    for line, length in zip(fused_lines, extracted_lengths):
        if length >= 100:
            filtered_lines.append(line)
            filtered_lengths.append(length)
    
    return filtered_lines, filtered_lengths

def RoughCrop(Line_coordinates, Original_img):
  # Input: Coordinates of the shortest line (calcualted in getLineLength) and 
  # Original images
  # Output: List of of images cropped by the size of the scale detected

  import cv2 as cv2
  import numpy as np
  import matplotlib.pyplot as plt
  
  list_of_crops = []
  for x in range(len(Original_img)):

      if Line_coordinates[x] != 0:
        
        try:
          img_gray = cv2.cvtColor(Original_img[x], cv2.COLOR_BGR2GRAY)
        except:
          img_gray = Original_img[x]
        
        TempCoor = Line_coordinates[x][0] # Assign the Coordinates to a value
        ## Cut the image we give 5% in x coordinate as buffer to detect the number
        
        height, width = img_gray.shape
        buffer = int(width*0.05)
  
        ## Check if the y value still lies within the range of the image
        if int(TempCoor[1] - buffer) > 0 and int(TempCoor[1] + buffer) < height:
          
          cropped_img = img_gray[TempCoor[1]-buffer:TempCoor[1]+buffer,TempCoor[0]:TempCoor[2]]
          
        else: ## If the image is not big enough/the scale too low we pad the size of the buffer
              ## In the unlikely case we also add it on the upper part
          padded_img = np.pad(img_gray, ((buffer, buffer), (0, 0)), mode='constant', constant_values= int(img_gray[0,0]))
          cropped_img = padded_img[TempCoor[1]:TempCoor[1] + 2*buffer,TempCoor[0]:TempCoor[2]]
        
        list_of_crops.append(cropped_img)
      else: 
        
        list_of_crops.append(Original_img[x])

  return(list_of_crops)

def detect_Number(List_with_Images):
  # Input Images, best case closely cropped to image
  # Output List of Numbers and letters 
  
  import easyocr
  reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

  Results = []
  for x in List_with_Images:
    try:
      result = reader.readtext(x, detail = 0) ## Predict
      Results.append(result)
    except:
      Results.append(0)
  
  return Results

def CropImage(Images):
  # takes as list of images
  # outputs a closley cropped list of images
  # based on CvCanny structure
  import cv2 as cv2
  import numpy as np
  counter = 0
  List_of_crops = []
  for x in Images:
    try:
      img = x
      
      Can = cv2.Canny(img,300,400, L2gradient = False)
      # The whiskers can confuse the process we have which is we already crop the far sides
      
      height, width = Can.shape
      Cropped_Can = Can[:, 10:width-10]
      Cropped_img_temp = img[:, 10:width-10]
      
      row_scan = np.sum(Cropped_Can, axis = 0)/255 ### Scans the image in rows
      coloumn_scan = np.sum(Cropped_Can, axis = 1)/255 ### Scans it in coloumns
      
      # Get the standard value for the scan
      reduced_column = coloumn_scan - coloumn_scan[0]
      
      # Find the index of the first number greater than 3 
      # This means we exclude all artifacts caused by jittery lines
      # smaller than 2 + 3 (5) Pixels
      first_index_y = np.argmax(reduced_column > 3)
    
      # Find the index of the last number greater than 3
      last_index_y = np.max(np.argwhere(reduced_column > 3))
      
      reduced_row = row_scan - row_scan[0]
      
      # Find the index of the first number greater than 3 
      # This means we exclude all artifacts caused by jittery lines
      # smaller than 2 + 3 (5) Pixels
      first_index_x = np.argmax(reduced_row > 3)
      
      # Find the index of the last number greater than 3
      last_index_x = np.max(np.argwhere(reduced_row > 3))
      
      if first_index_y-5 < 0: 
        first_index_y = 5
      if first_index_x-5 < 0: 
        first_index_x = 5
      
      #print(counter, first_index_x-5,last_index_x+5,first_index_y-5,last_index_y+5, img.shape)
      img_crop = Cropped_img_temp[first_index_y-5:last_index_y+5,first_index_x-5:last_index_x+5]
      counter += 1
      
      ## Very small and pixelated values have to be resized
      ## If the image is less than 100 pixels in any dimension
      ## We resize it with Inter cubic
      ## We resize according to the smaller axis
      
      height, width = img_crop.shape
      #print(img_crop.shape)
      ## Check size
      if (height < 100) or (width < 100):
        # Calculate scaling factors
        scale_height = 100/height
        scale_width = 100/width
        
        # Choose the bigger factor
        if scale_height > scale_width:
          scaling_factor = scale_height
        else:
          scaling_factor = scale_width
        #print(scaling_factor,height*scaling_factor)
        # Scale up
        img_crop = cv2.resize(img_crop, (int(width*scaling_factor),int(height*scaling_factor)), interpolation = cv2.INTER_CUBIC)
      
      List_of_crops.append(img_crop)
    except:
      List_of_crops.append(img)
      counter += 1
      
  return List_of_crops

def Sortlist(String_of_numbers):
  ### Drop entrys except numbers can cope with ['2','mm'] or ['2mm'] or ['2'], [] and floats.
  ### Input list of lists with strings containing numbers. 
  ### Output float values that are the digits found in string
  
  import re
  numbers_only = []
  ### Merge all substrings
  merged_list = [''.join(entry) for entry in String_of_numbers]
  
  ### First conver all values we can into floats
  ## Alternativ
  for y in merged_list:
    ### Drop everything that is "." or 0-9
    converted_list = [re.sub(r'[^0-9.]', '', value) for value in y]
    #### Try to convert strings to float
    #### If not possible empty entries add 0
    try:
      float_number = float(''.join(converted_list))
    except:
      float_number = 0.0

    ## We assume that every value over 10
    ## is micrometer so we divide by 1000
    mm_list = []
    
    if float_number > 10:
       numbers_only.append(float_number/1000)
    else:
        numbers_only.append(float_number)
        
  return numbers_only

def makeDfwithfactors(list_of_names, Scale_Mode,List_of_scale_numbers=[], list_of_lengths=[],filtered_lines =[], ConvFactor=0.002139):
  
  ### This function has two modes. 1) If the user declares that we only have one 
  ### scale we take the most common values of length and unit and 2) if more 
  ### than one exist we keep the list as they are.
  ### Then we enter the singular or mutliple values into the df
  
  import pandas as pd
  
  if Scale_Mode == 0:
    print(f"Using manual factor of {ConvFactor} px/mm")
    
    Scale_df = pd.DataFrame(list_of_names, columns=['image_name']) 
    Scale_df["distance_per_pixel"] = ConvFactor
    
    return Scale_df
  
  # Uniform Scale
  elif Scale_Mode == 1:
    LengthOpt = max(set(list_of_lengths), key=list_of_lengths.count)
    UnitOpt = max(set(List_of_scale_numbers), key=List_of_scale_numbers.count)
        
  # Different Scales
  elif Scale_Mode == 2:
    LengthOpt = list_of_lengths
    UnitOpt = List_of_scale_numbers
        
  else:
    print("No mode detected")  

  ## Note we add the individual lines to the scale mode who expects unifrom
  ## scales. This allows the user to see the stability of detection, but
  ## could cause confusion if looking at results.
  
  Scale_df = pd.DataFrame(list_of_names, columns =['Name'])

  Scale_df["metric_length"] = UnitOpt
  Scale_df["scale[px]"] = LengthOpt
  Scale_df["coordinates_scale"] = filtered_lines
  Scale_df["distance_per_pixel"] = Scale_df["metric_length"]/Scale_df["scale[px]"]
    
  return Scale_df

def DetectScale(DataDict,Scale_detector_mode=0,Conv_factor=0):
  ## Just give  the paths of image which are 
  Paths_of_Images = []
  Name_of_Images = []
  for image, annotations in DataDict.items():
  
          # Load image
          Paths_of_Images.append(annotations.get("image_path"))
          Name_of_Images.append(annotations.get("image_name"))
  
  if Scale_detector_mode != 0:
      
      Lengths, Line_Coors, List_of_images = getLineLength(Paths_of_Images)  ### Line lengths and lower right apart of image
      Rough_Images = RoughCrop(Line_Coors, List_of_images)                  ### Makes one number or list out of list(list(n,n1,n2), list(n,n1,n2),...)
      Small_Images = CropImage(Rough_Images)
      Detected_Numbers = detect_Number(Small_Images)
      Numbers = Sortlist(Detected_Numbers)
  
  else:
       ## Set an empty list to prevent, which is a dummy for scale mode 0
      Numbers = []
      Lengths = []
      Line_Coors = [] 
  
  ## Should all work until here
  ScaleDataframe = makeDfwithfactors(Name_of_Images,Scale_detector_mode,Numbers,Lengths,Conv_factor)
  
  return ScaleDataframe

