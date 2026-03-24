def Classify_Species(Folder_With_Images, Classifier_Location):
    import os
    import numpy as np
    from ultralytics import YOLO
    import re

    model = YOLO(Classifier_Location)
    
    image_paths = [os.path.join(Folder_With_Images, f) 
                   for f in os.listdir(Folder_With_Images) 
                   if f.lower().endswith(('.jpg', '.png'))]

    if not image_paths:
        return {}

    class_labels = model.names
    results_data = {}

    for image_path in image_paths:
        # Run inference at 1280 for maximum morphology detail
        result = model(image_path, imgsz=1280, verbose=False)[0]

        # result.probs.data is the tensor of probabilities
        probs = result.probs.data.cpu().numpy() 
        
        # Get index of the highest probability
        top_idx = np.argmax(probs)
        top_conf = probs[top_idx]

        # Create a dictionary of all class probabilities for this image
        prob_dict = {class_labels[i]: round(float(probs[i]), 4) for i in range(len(probs))}
        
        predicted_species = class_labels[top_idx] if top_conf >= 0.1 else "uncertain"

        filename = os.path.basename(image_path)
        filename = re.sub(r'_Daphnia\.jpg$', '.jpg', filename)

        # Store the species AND the full probability map
        results_data[filename] = {
            "prediction": predicted_species,
            "confidence": round(float(top_conf), 4),
            "all_probabilities": prob_dict
        }

    return results_data

if __name__ == "__main__":
    #Desastrous for pulex
    folder = r"C:\Users\hanss\Desktop\Test_Images_50_Subset\Images_subset\Pulex"
    classifier = r"C:\Users\hanss\Desktop\DaphniaDetectv2\src\daphniadetectv2\Model\classify\weights\best.pt"
    
    classifications = Classify_Species(folder, classifier)
    
    print(f"{'Filename':<30} | {'Prediction':<15} | {'Conf':<6} | {'Other Likely Candidates'}")
    print("-" * 100)
    
    for filename, data in classifications.items():
        # Filter for other classes with > 10% probability to see the "confusion"
        confusion = {k: v for k, v in data['all_probabilities'].items() if v > 0.1 and k != data['prediction']}
        
        print(f"{filename[:30]:<30} | {data['prediction']:<15} | {data['confidence']:<6} | {confusion}")