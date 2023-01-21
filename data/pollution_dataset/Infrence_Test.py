import os
import time
from matplotlib import pyplot as plt
import cv2
import torch
#######
import os
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torchvision.transforms as Transforms
import pandas as pd
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.patches as patches
# for image augmentations
#from albumentations.pytorch.transforms import ToTensorV2
import torchvision.transforms as T
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class_labels = {0: "GRAFFITI",
                1: "FADED_SIGNAGE",
                2: "POTHOLES",
                3: "GARBAGE",
                4: "CONSTRUCTION_ROAD",
                5: "BROKEN_SIGNAGE",
                6: "BAD_STREETLIGHT",
                7: "BAD_BILLBOARD",
                8: "SAND_ON_ROAD",
                9: "CLUTTER_SIDEWALK",
                10: "UNKEPT_FACADE"
                }

results = []
model.eval()

with torch.no_grad():
    images = list(torch.from_numpy(image).to(device) for image in images)

    output = model(images)
for i, im in enumerate(img):
    boxes = output[i]['boxes'].data.cpu().numpy()
    scores = output[i]['scores'].data.cpu().numpy()
    labels = output[i]['labels'].data.cpu().numpy()

    result = {
        'class': labels,
        'image_path': image_id[0],
        'name': class_labels[labels],
        'xmin': box[0],
        'ymin': box[1],
        'xmax': box[2],
        'ymax': box[3]}
    results.append(result)

df_sub = pd.DataFrame(results, columns=['class', 'image_path', 'name', 'xmin', 'ymin', 'xmax', 'ymax'])

df_sub['xmin'] = df_sub['xmin'] / 2

df_sub['ymin'] = df_sub['ymin'] / 2

df_sub['xmax'] = df_sub['xmax'] / 2

df_sub['ymax'] = df_sub['ymax'] / 2

df_sub = df_sub.astype({"xmax": int, "xmin": int, "ymax": int, "ymin": int})

df_sub.to_csv('DETR_WithOutAugmentation_1Epoch.csv', index=False)
df_sub.head()


if __name__ == '__main__':

    train_images_location=os.getcwd()+'/data/pollution_dataset/train/images/'
    train_labels_location=os.getcwd()+'/data/pollution_dataset/train/labels/'
    box_annotation=os.getcwd()+'/data/pollution_dataset/train.csv'

    test_res_folder=os.getcwd()+'runs/test/exp/labels/'
    test_images_location = os.getcwd() + '/data/pollution_dataset/test/images/'

    img_location=list(os.listdir(test_images_location))

    class_labels = {0: "GRAFFITI",
                1: "FADED_SIGNAGE",
                2: "POTHOLES",
                3: "GARBAGE",
                4: "CONSTRUCTION_ROAD",
                5: "BROKEN_SIGNAGE",
                6: "BAD_STREETLIGHT",
                7: "BAD_BILLBOARD",
                8: "SAND_ON_ROAD",
                9: "CLUTTER_SIDEWALK",
                10: "UNKEPT_FACADE"
                }

    results = []


    #with torch.no_grad():
    #    images = list(torch.from_numpy(image).to(device) for image in images)

    #output = model(images)
    for i, im in enumerate(img_location):
        masks = pd.read_csv(img_location)

        boxes = output[i]['boxes'].data.cpu().numpy()
        scores = output[i]['scores'].data.cpu().numpy()
        labels = output[i]['labels'].data.cpu().numpy()

        result = {
            'class': labels,
            'image_path': image_id[0],
            'name': class_labels[labels],
            'xmin': box[0],
            'ymin': box[1],
            'xmax': box[2],
            'ymax': box[3]}
        results.append(result)

df_sub = pd.DataFrame(results, columns=['class', 'image_path', 'name', 'xmin', 'ymin', 'xmax', 'ymax'])

df_sub['xmin'] = df_sub['xmin'] / 2

df_sub['ymin'] = df_sub['ymin'] / 2

df_sub['xmax'] = df_sub['xmax'] / 2

df_sub['ymax'] = df_sub['ymax'] / 2

df_sub = df_sub.astype({"xmax": int, "xmin": int, "ymax": int, "ymin": int})

df_sub.to_csv('DETR_WithOutAugmentation_1Epoch.csv', index=False)
df_sub.head()
