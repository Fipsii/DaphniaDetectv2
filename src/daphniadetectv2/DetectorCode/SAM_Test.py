import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO

# Define variables
source_dir = Path(r"C:\Users\hanss\Desktop\Daphnia\YOLO_Classification_Dataset_Save\cucullata I")
weights_path = r"C:\Users\hanss\weights.pt"
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
padding_px = 0  # Define padding in pixels

# Aggregate and randomly sample 50 image paths
all_images = [img for img in source_dir.rglob("*") if img.suffix.lower() in valid_extensions]
sample_size = min(50, len(all_images))
sampled_images = random.sample(all_images, sample_size)

# Initialize model
model = YOLO(weights_path)

# Initialize plot grid (5 rows, 10 columns)
fig, axes = plt.subplots(5, 10, figsize=(25, 12))
axes = axes.flatten()

# Execute inference and map to subplots
for i, img_path in enumerate(sampled_images):
    # Predict without disk I/O
    results = model.predict(source=str(img_path), save=False, verbose=False, imgsz=1024)
    result = results[0]
    
    # Generate annotated array
    annotated_img = result.plot()
    H, W = annotated_img.shape[:2]
    
    # Isolate highest confidence prediction and crop with padding
    if len(result.boxes) > 0:
        max_conf_idx = result.boxes.conf.argmax().item()
        box = result.boxes.xyxy[max_conf_idx].cpu().numpy().astype(int)
        
        # Calculate padded coordinates clamped to array boundaries
        x1 = max(0, box[0] - padding_px)
        y1 = max(0, box[1] - padding_px)
        x2 = min(W, box[2] + padding_px)
        y2 = min(H, box[3] + padding_px)
        
        display_img = annotated_img[y1:y2, x1:x2]
    else:
        # Fallback to full image if no detection occurs
        display_img = annotated_img

    # Convert BGR to RGB for matplotlib
    display_img_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
    
    # Display array on corresponding axis
    axes[i].imshow(display_img_rgb)
    axes[i].axis("off")

# Clear unused axes if the directory contains fewer than 50 valid images
for j in range(i + 1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.show()