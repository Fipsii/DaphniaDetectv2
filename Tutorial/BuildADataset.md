# Create your on model

Microscope images can be wildly different depended on the zoom and microscope, so it may be the case that our models do not apply well onto your research data or you want to detect and measure different body parts that we have not provided in our detection models. For these cases it might be necessary to either A) make a whole new model or B) train our baseline models onto your data.

For both cases you first need to gather and annotate data for YOLO this can be done in multiple annoation programs like CVAT or Roboflow. Plug in your data and annotate your data the way you need and export it as YOLOv11 annotations, which should leave you with a folder structured like this:

```
/dataset
  /train
    /images
      img001.jpg
      img002.jpg
      …
    /labels
      img001.txt
      img002.txt
      …
  /val
    /images
      img003.jpg
      img004.jpg
      …
    /labels
      img003.txt
      img004.txt
      …
```
Some labelling programs only output the labels in which case you need to create this folder structure on your own. Now your data is missing only header file that you should place in the same folder called data.yaml. This file looks different depending on the task, but the basic makeup is the same:
You can find the data.yaml files we used for segmentation, classifaction and box detection in the tutorial folder in this github.

Example for simple classification:
```
path: path/to/your/folder (The folder in which your data.yaml lies)
train: train (The path from your folder to the train/val/test data)
val: val
test: test

names:
  0: cucullata
  1: longicephala
  2: longispina
  3: magna
  4: pulex

```

Once this is done your data set is created. However you can do way more before that to your data like augmentation, setting weights or other tricks to increase your models performance. For this visit the offical [Ultralytics YOLO Docs](https://docs.ultralytics.com)
