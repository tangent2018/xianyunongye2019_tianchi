#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage
import imgaug

ROOT_DIR = os.path.abspath('./')

from mrcnn.config import Config
from mrcnn import utils as mrcnnUtils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import utils

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)
    
# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    mrcnnUtils.download_trained_weights(COCO_MODEL_PATH)    

#%%
# Configurations
config = utils.XYNYConfig()
config.display()

#%%
class XYNYDataset(mrcnnUtils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_image_and_labelmap_paths(self, all_image_paths):
        """Generate image_info by image_paths and labelmap_paths.
        all_image_paths: list of image_path and labelmap_paths.
        """
        # Add classes
        self.add_class("XYNY", 1, "tabacoo")
        self.add_class("XYNY", 2, "corn")
        self.add_class("XYNY", 3, "coix")
        
        # Add images
        for i in range(len(all_image_paths)):
            image_path = all_image_paths[i][0]
            labelmap_path = all_image_paths[i][1]
            LMB_path = image_path.split('.')[0]+'_LMB.png'
            self.add_image("XYNY", image_id=i, path=image_path, labelmap_path=labelmap_path, LMB_path=LMB_path)
    
    def load_mask(self, image_id):
        # Load image
        labelmap_path = self.image_info[image_id]['labelmap_path']
        mask, class_ids = utils.create_mask_and_class_ids(labelmap_path)
        return mask.astype('bool'), class_ids
    
    def load_LMB(self, image_id):
        LMB_path = self.image_info[image_id]['LMB_path']
        LMB = skimage.io.imread(LMB_path)
        return LMB

#%%
# generate dataset_train and dataset_val
# dataset_train:0/256/512/768pad; dataset_test:0pad 50%

# generate train_data
file_name_list = [
    '..\\data\\extract_data\\train_data_0pad_has_class.txt', '..\\data\\extract_data\\train_data_256pad_has_class.txt',
    '..\\data\\extract_data\\train_data_512pad_has_class.txt', '..\\data\\extract_data\\train_data_768pad_has_class.txt'
]

train_data = []
for file_name in file_name_list:
    with open(file_name, 'r') as f:
        f_lines = f.readlines()
    image_paths = []
    labelmap_paths = []
    for content in f_lines:
        image_path, _, _, _, labelmap_path = content.strip().split(',')
        image_paths.append(image_path)
        labelmap_paths.append(labelmap_path)
    all_image_paths = list(zip(image_paths, labelmap_paths))
    random.shuffle(all_image_paths)
    train_data.extend(all_image_paths)

# generate val_data
file_name = file_name_list[0]
with open(file_name, 'r') as f:
    f_lines = f.readlines()
image_paths = []
labelmap_paths = []
for content in f_lines:
    image_path, _, _, _, labelmap_path = content.strip().split(',')
    image_paths.append(image_path)
    labelmap_paths.append(labelmap_path)
all_image_paths = list(zip(image_paths, labelmap_paths))
random.shuffle(all_image_paths)
data_slice = int(len(all_image_paths)*0.5)
val_data = all_image_paths[data_slice:]
    
print ('train_data num:', len(train_data), ', val_data num:', len(val_data))
    
# Training dataset
dataset_train = XYNYDataset()
dataset_train.load_image_and_labelmap_paths(train_data)
dataset_train.prepare()

# Validation dataset
dataset_val = XYNYDataset()
dataset_val.load_image_and_labelmap_paths(val_data)
dataset_val.prepare()

#%%
# Create Model
# 仅初始化时运行
# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
# 以下中途训练时使用
#model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
#model_path = model.find_last()
#print("Loading weights from ", model_path)
#model.load_weights(model_path, by_name=True)


# 数据增强
augmentation = imgaug.augmenters.Sometimes(p = 0.5, 
                    # 1）50%左右翻转；2）50%上下翻转；3）crop
                    then_list = [imgaug.augmenters.Fliplr(.5), imgaug.augmenters.Flipud(.5), imgaug.augmenters.CropAndPad(percent=(0, 0.1))],
                    # 1）旋转0，90，180，270；2）crop
                    else_list = [imgaug.augmenters.Rot90((0,3)), imgaug.augmenters.CropAndPad(percent=(0, 0.1))])
# lr=1e-4 训练28个epoch
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=28, 
            layers="all",
            augmentation=augmentation)

# lr=3e-5 再训练10个epoch
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 30,
            epochs=38, 
            layers="all",
            augmentation=augmentation)

