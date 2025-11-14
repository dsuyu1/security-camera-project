![img](https://github.com/dsuyu1/security-camera-project/blob/main/ztf_workflow.png?raw=true)

# Privacy-First CCTV System
Making a new and secure security camera framework for my senior project class.


# How to train YOLO
Download the dataset from https://www.kaggle.com/datasets/iamprateek/wider-face-a-face-detection-dataset?resource=download
Place the dataset into the repo and run the convert_to_yolo.py script.
There may be file pathing errors so ensure the dataset paths are as described in the python script editing either the path used in the python file or of the dataset.
From here you can download your yolo versions weights from ultralytics (https://docs.ultralytics.com/models/yolo12/#detection-performance-coco-val2017) and run the following.
yolo train model=\<your model weights\> data=face.yaml imgsz=640 epochs=100 batch=16 