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

    # Run model prediction on all images at once
    results = model(image_paths,batch=1, stream = True,imgsz=1280, verbose=False)  # Batch inference

    # Class mapping
    class_labels = model.names

    # Process predictions
    results_data = {}
    for image_path, result in zip(image_paths, results):
        probs = result.probs.cpu().numpy()  # Convert to NumPy array

        predicted_class = np.argmax(probs.data) if np.max(probs.data) >= 0.75 else np.nan
        species = class_labels.get(predicted_class, "unknown")  # Get species name

        filename = os.path.basename(image_path)  # Extract filename
        results_data[filename] = species  # Store result in dictionary

    return results_data  # Return dictionary instead of DataFrame


