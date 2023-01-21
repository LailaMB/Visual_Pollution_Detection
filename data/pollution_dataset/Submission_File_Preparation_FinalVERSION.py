from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

if __name__ == '__main__':


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
    pred_label_location = os.getcwd() + '/runs/test/exp7/labels/'
    df_test = pd.read_csv(os.getcwd()+"/data//pollution_dataset/test.csv")

    test_images=os.listdir(test_images_location)
    pred_label_file=os.listdir(pred_label_location)


    def txt_to_csv(single_object, scaled):

        PRED_PATH = pred_label_location

        results = []

        for file in os.listdir(PRED_PATH):
            # result = ''
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
                            xmin=np.round(xmin/2)
                            ymin = np.round(ymin / 2)
                            xmax = np.round(xmax / 2)
                            ymax = np.round(ymax / 2)

                        result = {
                            'class': class_num,
                            'image_path': file.split('.')[0] + '.jpg',
                            'name': class_labels[int(class_num)],
                            'xmin': xmin,
                            'ymin': ymin,
                            'xmax': xmax,
                            'ymax': ymax,
                            'confidence': confidence}

                        img_results.append(result)

                    if (single_object):
                        sorted_img_results = sorted(img_results, key=lambda d: d['confidence'], reverse=True)
                        results.append(sorted_img_results[0])
                    else:
                        results.extend(img_results)

        return results


    ###### function starts here....
    results = txt_to_csv(single_object=True, scaled=True)



    ######## IMOPRTANT....
    new_hight=int(1080)
    new_width=int(1920)



    #for i, row in df_test.iterrows():
    for i in range(0, len(df_test)):

        print(i)

        objects = list(filter(lambda obj: obj['image_path'] == df_test.iloc[i]['image_path'], results))

        if objects != []:
            for j, _ in enumerate(objects):
                df_test.at[i, 'name'] = objects[j]['name']
                df_test.at[i, 'class'] = objects[j]['class']
                df_test.at[i, 'xmin'] = int(objects[j]['xmin'])
                if objects[j]['xmin']==0:
                    df_test.at[i, 'xmin'] = int(df_test.at[i, 'xmin'])+1
                df_test.at[i, 'xmax'] = int(objects[j]['xmax'])
                df_test.at[i, 'ymin'] = int(objects[j]['ymin'])
                if df_test.at[i, 'ymin']==0:
                    df_test.at[i, 'ymin']=int(df_test.at[i, 'ymin'])+1
                df_test.at[i, 'ymax'] = int(objects[j]['ymax'])

                


        #else:
        #    df_test.at[i, 'class'] = -1
        #    df_test.at[i, 'name'] = "-1"
        #    df_test.at[i, 'xmin'] = -1
        #    df_test.at[i, 'xmax'] = -1
        #    df_test.at[i, 'ymin'] = -1
        #    df_test.at[i, 'ymax'] = -1

    ##### WITHOUT NORMALIZATION
    #df_test = df_test.reindex(columns=['class', 'image_path', 'name', 'xmax', 'xmin', 'ymax', 'ymin'])
    df_test.to_csv('AJCAS_KSU_Submission_20Jan_800_Epcoh119_Final.csv', index=False)

