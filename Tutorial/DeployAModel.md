# Deploy your Model

Now that you have a finished model you can add it to the pipeline!
You can see how we do it based on the classification model deployment.

Here a slightly adapt version highlighting the curcial steps:

You load the model similarly to the validation and testing steps.
We also create a list of all the images we want to have detect, you can
decide if its already a cropped image by specifying the path of Folder_With_Images.
Class_labels helps use to translate the predictions from numbers to more informative strings, this not strictly necessary.

```
def Classify_Species(Folder_With_Images, Classifier_Location):

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
```

No that we have initiated a model we can predict on each image in a loop. Technically it is possible to input a whole folder as image_path and not loop. however depending on your system this can lead to a RAM shortage, which is why we avoid this.
We set verbose false to suppress the loading print of each image.

We convert the results into an array to extract the data and make our inference. we customize the output and designate any result with a lower probability of 75% as unsure indicating that the model is not confident enough. Where this threshold should be dependence on your model an the resilience against false positives.
To be able to merge this data later into the results data frame we make a dicit with the image name and the detect species.
```
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
```
