from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tqdm

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':

    ####### Confidence threshold
    set_confidence_threshold=0.4

    class_labels=["GRAFFITI", "FADED_SIGNAGE", "POTHOLES", "GARBAGE",
             "CONSTRUCTION_ROAD","BROKEN_SIGNAGE", "BAD_STREETLIGHT", "BAD_BILLBOARD", "SAND_ON_ROAD", "CLUTTER_SIDEWALK", "UNKEPT_FACADE"]


    #train_images_location=os.getcwd()+'/data/pollution_dataset/train/images/'
    #train_labels_location=os.getcwd()+'/data/pollution_dataset/train/labels/'
    #box_annotation=os.getcwd()+'/data/pollution_dataset/train.csv'



    ####Format: [x_min, y_min, x_max, y_max]
    def yolo_to_voc(x, y, w,h, width, height):
        xmax = int((x * width) + (w * width) / 2.0)
        xmin = int((x * width) - (w * width) / 2.0)
        ymax = int((y * height) + (h * height) / 2.0)
        ymin = int((y * height) - (h * height) / 2.0)

        return (xmin, ymin, xmax, ymax)

    ############################### Generate Bounding Boxex
    ### Read predicted label
    ### locate test images in the predicted folder
    from numpy import loadtxt
    from csv import writer


    test_images_location = os.getcwd() + '/data/pollution_dataset/test/images/'
    pred_label_location = os.getcwd() + '/runs/test/exp10/labels/'
    df_test = pd.read_csv(os.getcwd()+"/data//pollution_dataset/test.csv")

    test_images=os.listdir(test_images_location)
    pred_label_file=os.listdir(pred_label_location)


    def txt_to_csv(scaled, conf_threshold=0.5):
        PRED_PATH = pred_label_location

        results = []

        for file in os.listdir(PRED_PATH):
            if file.endswith('.txt'):
                file_path = f"{PRED_PATH}/{file}"
                with open(file_path, 'r') as f:

                    img_results = []

                    for line in f:
                        result = line.rstrip('\n')
                        class_num = result.split(' ')[0]

                        # Make format conversion here
                        xmin = float(result.split(' ')[1])
                        ymin = float(result.split(' ')[2])
                        xmax = float(result.split(' ')[3])
                        ymax = float(result.split(' ')[4])

                        confidence = float(result.split(' ')[5])

                        if (scaled):
                            xmin, ymin, xmax, ymax = yolo_to_voc(xmin, ymin, xmax, ymax, width=1920, height=1080)
                            xmin = np.round(xmin / 2)
                            ymin = np.round(ymin / 2)
                            xmax = np.round(xmax / 2)
                            ymax = np.round(ymax / 2)


                        result = {
                            'class': class_num,
                            'image_path': file.split('.')[0] + '.jpg',
                            'name': class_labels[int(class_num)],
                            'xmax': xmax,
                            'xmin': xmin,
                            'ymax': ymax,
                            'ymin': ymin,
                             'confidence': confidence}

                        img_results.append(result)

                    # filter objects with confidence > conf_threshold
                    filtered_results = [obj for obj in img_results if obj['confidence'] >= conf_threshold]

                    if filtered_results == []:
                        # if confidence of all objects < conf_threshold consider all boxes
                        sorted_img_results = sorted(img_results, key=lambda d: d['confidence'], reverse=True)
                        results.append(sorted_img_results[0])
                    #    print(results)
                        #results.extend(img_results)

                    else:

                        results.extend(filtered_results)

        return results


    results = txt_to_csv(scaled=True, conf_threshold=set_confidence_threshold)

    df_test = pd.read_csv(os.getcwd()+"/data//pollution_dataset/test.csv")
    df_pred = pd.DataFrame()

    for i in range(len(df_test)):
        print(i)
        boxes = list(filter(lambda j: j['image_path'] == df_test.iloc[i]['image_path'], results))
        #print(boxes)

        for b in boxes:
                df_pred = df_pred.append(b, ignore_index=True)

        if boxes==[]:
            print(df_test.iloc[i]['image_path'])
            b = {
                'class': -1,
                'image_path': df_test.iloc[i]['image_path'],
                'name': -1,
                'xmax': -1,
                'xmin':  -1,
                'ymax': -1,
                'ymin': -1,
                'confidence': -1}
            df_pred=df_pred.append(b, ignore_index=True)



    df_pred.drop(['confidence'], axis=1, inplace=True)


    ##### WITHOUT NORMALIZATION
    df_pred.to_csv('AJCAS_KSU_Submission_21Jan_800_multi_Threshold_04_exp10_Focal.csv', index=False)

