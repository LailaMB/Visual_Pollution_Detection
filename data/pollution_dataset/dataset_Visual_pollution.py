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

class VisonPollution(torch.utils.data.Dataset):
    def __init__(self, root,root_csv,width, height, transforms=None):
        self.root = root
        self.root_csv=root_csv
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(os.listdir(self.root))
        self.masks = pd.read_csv(self.root_csv)
        self.classes=['back','GARBAGE','BAD_BILLBOARD','SAND_ON_ROAD','GRAFFITI','POTHOLES','CLUTTER_SIDEWALK','CONSTRUCTION_ROAD',
                     'UNKEPT_FACADE']
        self.height = height
        self.width = width

    def __getitem__(self, idx):
        # load images and masks
        #img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        img = cv2.imread(self.root + self.imgs[idx])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        #img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # cv2 image gives size as height x width
        img_res=img_rgb
        wt = img.shape[1]
        ht = img.shape[0]

        boxes_norm = []
        boxes_orig=[]
        labels = []
        #print(self.root + self.imgs[idx])

        for member in range(len(self.masks)):

            if self.imgs[idx] in self.masks.image_path[member]:

                labels.append(self.masks.label[member])
                # bounding box
                xmin = 2*int(self.masks.xmin[member])
                xmax = 2*int(self.masks.xmax[member])
                ymin = 2*int(self.masks.ymin[member])
                ymax = 2*int(self.masks.ymax[member])
                xmin_corr = (xmin / wt) * self.width
                xmax_corr = (xmax / wt) * self.width
                ymin_corr = (ymin / ht) * self.height
                ymax_corr = (ymax / ht) * self.height
                boxes_norm.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])
                boxes_orig.append([xmin, ymin, xmax, ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes_norm, dtype=torch.float32)
        # getting the areas of the boxes
        area=0
        if len(boxes)>0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["name"] = self.imgs[idx]
        target["boxes_orig"] = boxes_orig
        target["boxes_norm"] = boxes_norm
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        # image_id
        image_id = torch.tensor([idx])
        #target["image_id"] = image_id
        target['image']=img_res

        #if self.transforms:
        #    sample = self.transforms(image=img_res,
        #              bboxes=target['boxes'],
        #              labels=labels)

        #    img_res = sample['image']
        #    target['boxes'] = torch.Tensor(sample['bboxes'])

        return target

    def __len__(self):
        return len(self.imgs)



def plot_img_bbox(img, target):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1,1)
    fig.set_size_inches(5,5)
    a.imshow(img)
    for box in (target['boxes_orig']):
        x, y, width, height  = box[0], box[1], box[2]-box[0], box[3]-box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth = 2,
                                 edgecolor = 'r',
                                 facecolor = 'none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()

def get_object_detection_model(num_classes):

        # load a model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        return model

if __name__ == '__main__':


    train_images_location=os.getcwd()+'/data/pollution_dataset/train/images/'
    train_labels_location=os.getcwd()+'/data/pollution_dataset/train/labels/'

    test_images_location = os.getcwd() + '/data/pollution_dataset/test/images/'
    box_annotation=os.getcwd()+'/data/pollution_dataset/train.csv'



    ####Format: [x_min, y_min, x_max, y_max]
    def pascal_voc_to_coco(xmin, ymin, xmax, ymax):
        return [xmin, ymin, xmax - xmin, ymax - ymin]



    ####Format: [x_min, y_min, x_max, y_max]
    def pascal_voc_to_yolo(xmin, ymin, xmax, ymax, image_w, image_h):

        return [((xmax + xmin) / (2 * image_w)), ((ymax + ymin) / (2 * image_h)), (xmax - xmin) / image_w,
                (ymax - ymin) / image_h]


    imgs = list(os.listdir(train_images_location))

    image_w=1920
    image_h=1080

    masks = pd.read_csv(box_annotation)

    ### check name to imagepath......
    #### generate .txt files with the sae name of images and insert them in Label folder
    ##### 3 0.5 0.4 0.78 0.78

    Ostreams = {}

    for img_file in imgs:
        print(img_file)
        os.system("echo. > " + train_labels_location + "/" + img_file[:-4] + ".txt")
        open(train_labels_location + "/" + img_file[:-4] + ".txt", 'w').close()
        Ostreams[img_file] = open(train_labels_location + "/" + img_file[:-4] + ".txt", 'a')

    for Index in range(len(masks)):
        print(Index)
        # Order = xmax xmin ymax ymin
        img_label = masks.label[Index]
        img_name = masks.image_path[Index]
        xmax = 2 * masks.xmax[Index]
        xmin = 2 * masks.xmin[Index]
        ymax = 2 * masks.ymax[Index]
        ymin = 2 * masks.ymin[Index]

        if (xmin<0):
            xmin=0
            print('HIIIII')
        if (ymin < 0):
            ymin = 0
        if (ymax >1080):
            print('HIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII')
            ymax = 1080
        if (xmax > 1920):
            xmax = 1920




        fix_coords = pascal_voc_to_yolo(xmin, ymin, xmax, ymax, image_w, image_h)

        Ostreams[img_name].write(str(img_label) + " ")
        Ostreams[img_name].write(str(fix_coords[0]) + " ")
        Ostreams[img_name].write(str(fix_coords[1]) + " ")
        Ostreams[img_name].write(str(fix_coords[2]) + " ")
        Ostreams[img_name].write(str(fix_coords[3]) + "\n")

    for Stream in Ostreams:
        Ostreams[Stream].close()


    ############################### Generate Bounding Boxex
    ### Read predicted label
    









