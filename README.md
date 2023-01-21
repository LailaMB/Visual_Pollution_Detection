# AICAS_KSU Team: Smartathon Visual Pollution Detection

![Visual Pollution Detection](image.png)


## Dataset


The dataset is provided by the [Smartathon](https://smartathon.hackerearth.com). It consists of 9966 images with different visual pollution types: GRAFFITI, FADED SIGNAGE, POTHOLES, GARBAGE, CONSTRUCTION ROAD, BROKEN_SIGNAGE, BAD STREETLIGHT, BAD BILLBOARD, SAND ON ROAD, CLUTTER_SIDEWALK, UNKEPT_FACADE.

## Demo
- The model is provided as an integrated [Huggingface Space 🤗](https://huggingface.co/spaces/LailaMB/visual_pollution_detection) web deme [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LailaMB/visual_pollution_detection)
- The model is provided as a colab notebook <a href="https://colab.research.google.com/drive/1FFCN5_IxZ1mwb56twUvjuMvGUbX4VNu4?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

## Train
<code>python train_aux.py --workers 0 --device 0 --batch-size 8 --data data/pollution_dataset/visionpollution.yaml --img 800 800 --cfg cfg/training/yolo7-e6.yaml  --weights yolov7-e6_training.pt --name yolov7-pollution --hyp data/hyp.scratchratch.p.yaml</code>

## Test
<code>python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val </code>

## Inference
<code>python detect.py --weights best.pt --conf 0.3 --img-size 960 --source "IMAGE_PATH" </code>


## AICAS_KSU Team

Dr.Hebah ElGibreen, Prof. Yakoub Bazi, Dr. Mohamad Al Rahhal, Eng. Laila Bashmal
