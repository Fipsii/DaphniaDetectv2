# Train your own model

Now that you have created your own dataset you can train a model on it!
The basic loop of it is very easy thanks to YOLO. There are a few options you should consider even in a basic model.

First select the fitting model, based on data you trained you can find a full list of the YOLOv11 supported models [here](https://docs.ultralytics.com/models/yolo11/#supported-tasks-and-modes). If you want to use on of our models use the .pt files we have in our model folders. Doing this utilizes the already learned inforamtion about the task and complements it with your data.

Now you need to decide the model size. Generally bigger models are more accurate but need more computational power. 
Then you decide how long to train, setting the epochs. Set these low first and increase as you get a feeling for your model. 
After that set the Image size, YOLO will automatically resize your images to squares this size, increasing the size can be benefital for images with small features, but it will increase the computational demand drastically. 
Normally you can keep batches at 16 or -1 (auto-detect), to see what this actually means and which settings you can select you can again visit the YOLO docs.


```
from ultralytics import YOLO

# load pretrained YOLO model
model = YOLO("yolo11n.pt")  

# train
model.train(
    data="data.yaml",  # dataset config
    epochs=50,
    imgsz=640,
    batch=16
)
```

After setting the training loop, your PC will be busy for a while. Once finished a saved file with your model will be created.
You can validate your data using

```
metrics = model.val(data="data.yaml")
```

If your model seems to work well it is time to verfiy the model on your test data, using:

```
# Get the metrics like MAP
metrics = model.val(data="data.yaml", split="test")

# Get a visualized result
results = model.predict(source="/path/to/dataset/images/test", save=True)
```

Be careful how you deal with your test data once you verify your data on your test set the test set is used. If decide the performance is lacking and retrain the model based on this performance using the same test set your relegate it to a second validation set and not necessarly improve a fit.


