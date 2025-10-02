# Train your own model

Now that you have created your own dataset you can train a model on it!
The basic loop of it is very easy thanks to YOLO. There are a few options you should consider even in a basic model.

First select the fitting model, based on data you trained you can find a full list of the YOLOv11 supported models [here](https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes)
First is the model size

```
from ultralytics import YOLO

# load pretrained YOLO model
model = YOLO("yolo11n.pt")  

# train
model.train(
    data="data.yaml",  # dataset config
    epochs=50,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```
