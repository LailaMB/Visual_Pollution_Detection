# Smartathon Visual Pollution Detection

![Visual Pollution Detection](image.png)


## Dataset

The dataset is provided by the [Smartathon](https://smartathon.hackerearth.com) of images with different visual pollution types: (GRAFFITI, FADED SIGNAGE, POTHOLES, GARBAGE, CONSTRUCTION ROAD, BROKEN_SIGNAGE, BAD STREETLIGHT, BAD BILLBOARD, SAND ON ROAD, CLUTTER_SIDEWALK, UNKEPT_FACADE).


## Train
<code>python train_aux.py --workers 0 --device 0 --batch-size 8 --data data/pollution_dataset/visionpollution.yaml --img 800 800 --cfg cfg/training/yolo7-e6.yaml  --weights yolov7-e6_training.pt --name yolov7-pollution --hyp data/hyp.scratchratch.p.yaml</code>

## Test
<code>python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val </code>

## Inference
<code>python detect.py --weights best.pt --conf 0.3 --img-size 960 --source "IMAGE_PATH" </code>


## Demo
- Integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces/LailaMB/visual_pollution_detection). Try out the Web Demo [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/LailaMB/visual_pollution_detection)
- Implemented as colab notebook <a href="https://colab.research.google.com/drive/1FFCN5_IxZ1mwb56twUvjuMvGUbX4VNu4?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

git remote add origin https://github.com/LailaMB/Smartathon_Visual_Pollution_Detection.git
git remote add origin https://github.com/LailaMB/Smartathon_Visual_Pollution_Detection.git

git remote set-url origin https://github.com/LailaMB/Smartathon_Visual_Pollution_Detection.git