def Classify_Species(Folder_With_Images, Classifier_Location):
    import os
    import numpy as np
    from ultralytics import YOLO

    # Load the YOLO model once
    model = YOLO(Classifier_Location)

    # Get all image paths directly from the folder
    image_paths = [os.path.join(Folder_With_Images, filename) 
                   for filename in os.listdir(Folder_With_Images) 
                   if filename.endswith(('.jpg', '.png'))]  # Only process images

    if not image_paths:
        print("No images found in the specified folder.")
        return {}

    # Class mapping
    class_labels = model.names

    # Process predictions one at a time
    results_data = {}
    for image_path in image_paths:
        result = model(image_path, imgsz=1280, verbose=False)[0]  # Run inference on a single image

        probs = result.probs.cpu().numpy()  # Convert to NumPy array

        predicted_class = np.argmax(probs.data) if np.max(probs.data) >= 0.75 else np.nan
        species = class_labels.get(predicted_class, "unknown")  # Get species name

        filename = os.path.basename(image_path)  # Extract filename
        filename = re.sub(r'_Daphnia\.jpg$', '.jpg', filename) # Correct so it fits for the original image
        results_data[filename] = species  # Store result in dictionary

    return results_data  # Return dictionary instead of DataFrame

