# 📷Final Project | Computer Vision 📷
---
**Master's in Machine Learning for Health** | 2023-2024

Date: May 26th 2024

| Authors                   |       NIA         |
| ------------------------- |    -------------  |
| Alexia Durán Vizcaíno     |       429771      |
| Antonio Sánchez Salgado   |       428745      |

Instructor: Iván González Díaz

---
<center><img src='https://c0.piktochart.com/v2/uploads/ff026532-700a-40da-a157-d9a9281918f2/3e0101f1dde2efcf0c3bc8a76fdd32cff2995805_original.png' width=400 /></center>

---

## *Object Detection with Ultralytics **YOLOv8***

We will use YOLOv8 from Ultralytics as an alternative to Faster-RCNN with ResNet-50.

### 1. Load the PASCAL VOC dataset 

The function to load the database is defined below. To simplify its treatment, the database is provided in the form of images (`images`) and masks for each of the object instances (`instances`). Thus, for each image there are as many masks as there are objects in it (with the segmentation of each one of them). Also, the category of each object is encoded in the name of the image. In addition, the `classes` semantic segmentation and the `masks` instance segmentation are provided.

### 2. Convert to appropriate YOLOv8 format 

The Ultralytics YOLO format is a dataset configuration format that allows you to define the dataset root directory, the relative paths to training/validation/testing image directories or *.txt files containing image paths, and a dictionary of class names.  
Labels for this format should be exported to YOLO format with one *.txt file per image. If there are no objects in an image, no *.txt file is required. The *.txt file should be formatted with one row per object in `class x_center y_center width height` format. Box coordinates must be in normalized xywh format (from 0 to 1). If your boxes are in pixels, you should divide `x_center` and `width` by image width, and `y_center` and `height` by image height. Class numbers should be zero-indexed.

### 3. Fine-tuning YOLOv8

Now, we use a pre-trained model of YOLOv8 for object detection and we finetune it in our specific dataset. We first need to create the `yolov8_config.yaml` file which contains the paths for the training and test partitions, and the information related to the number of classes that it needs to detect. Furthermore, we can define some other params of the model, such as the predefined image size (`imgsz=640`) or the optimizer (`optimizer='SGD'`).

### 4. Evaluate the model
![Evaluation](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/detect/confusion_matrix.png)
### 5. Save the results
![Results](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/detect/SGDe50/results.png)

### 6. Visualize the results

After using the model to predict over the test dataset, we can find all the images in `./runs/detect/predict`.   
Below, we print a combination of these test images.

![Sample Detection](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/detect/SGDe50/val_batch2_pred.jpg)

### 7. Inference with "micasa images"

Finally, we wanted to experiment if our model was able to detect the objects with other unseen pictures. For this purposes, we have made some home-made photos in order to test our resulting model. Here, we show the results.

![Sample Detection](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/detect/micasa_predict/IMG_6157.jpg)

## *Semantic Segmentation with Ultralytics **YOLOv8***

### 1. Create polygon from masks

In the PASCAL VOC dataset, we have for every object detected a mask of the object and the class in an `instance` image. Similarly to the preparation of the dataset in the object detection task, we now need to get this mask images into `labels.txt` to input our model. In this case, we need a unique .txt file per image, in which each object information is in a different row. Then, each row must have the object `class`, a number representing the class of the object, and the `polygon`, the bounding coordinates around the mask area, normalized to be between 0 and 1. 

### 2. Fine-tuning YOLOv8-seg

### 3. Evaluate 
![Evaluation](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/segment/confusion_matrix.png)

### 4. Save the results
![Results](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/segment/seg-SGDe50/results.png)
### 5. Visualize the results

After using the model to predict over the test dataset, we can find all the images in `./runs/segment/predict`.   
Below, we print a combination of these test images.
![Sample Detection](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/segment/seg-SGDe50/val_batch2_pred.jpg)

### 6. Inference with "micasa images"

Finally, we wanted to experiment if our model was able to detect the objects with other unseen pictures. For this purposes, we have made some home-made photos in order to test our resulting model. Here, we show the results.
![Sample Detection](https://github.com/antoniosanch3/YOLOv8-Project/blob/main/runs/segment/micasa_predict/IMG_6157.jpg)
