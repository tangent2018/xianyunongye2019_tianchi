# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 14:16:54 2019

@author: l
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
#import keras
import sys
import PIL
from PIL import Image
import re
import random
import skimage

from mrcnn import utils as mrcnnUtils
from mrcnn.config import Config

#==============================================================================
# DATA PROCESS
#==============================================================================

COLOR_MAP = np.array([
        [0,0,0], #0 BG 其他
        [255,0,0], #1 tabacco 烤烟 红
        [255,255,0], #2 corn 玉米 黄
        [255,128,0], #3 coix 薏仁米 橙
        [0,255,0] #4 with_white 白色物 绿
    ])

def apply_label_map(image, label_map, color_map=COLOR_MAP, alpha=0.5):
    """Apply the given label_map to the image.
    """
    image_show = image.copy()
    if len(label_map.shape)>2:
        label_map = label_map[:,:,0]
    for label in np.unique(label_map):
        if label == 0:
            continue
        color = color_map[label]
        for c in range(3):
            image_show[:, :, c] = np.where(label_map == label,
                                      image_show[:, :, c] *
                                      (1 - alpha) + alpha * color[c],
                                      image_show[:, :, c])
    return image_show

# =============================================================================
# DATASET
# =============================================================================

def show_image_labelmap(image, label_map):
    image_show = apply_label_map(image, label_map)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
    ax1.imshow(image)
    ax2.imshow(image_show)
    plt.show()

def divide_mask(labelmap):
    # 将不同的label分离
    class_ids = np.unique(labelmap)
    class_ids = list(class_ids)
    if 0 in class_ids:
        class_ids.remove(0)
    mask_frame = np.zeros_like(labelmap)
    masks = []
    for i in range(len(class_ids)):
        mask = mask_frame.copy()
        mask = np.where(labelmap==class_ids[i], 1, 0)
        masks.append(mask)
    return np.array(masks), np.array(class_ids)

def combine_bboxs(bboxs):
    '''
    将bbox中互相包含的整合为一个，整合标准：只要有重叠，就整合为一个更大的bbox
    input:
    bboxs: list of [x1, y1, x2, y2]    
    output:
    bboxs: list of [x1, y1, x2, y2] which has been combined
    '''
    remove_list = []
    for i in range(len(bboxs)):
        for j in range(i+1, len(bboxs)):
            xi1, yi1, xi2, yi2 = bboxs[i]
            xj1, yj1, xj2, yj2 = bboxs[j]
            # 分别判断：
            # (xi2, yi2)在bboxs[j]里面
            # (xi1, yi1)在bboxs[j]里面
            # (xj2, yj2)在bboxs[i]里面
            # (xj1, yj1)在bboxs[i]里面
            if ((xi2>=xj1 and yi2>=yj1) and (xi2<=xj2 and yi2<=yj2)) or \
               ((xi1>=xj1 and yi1>=yj1) and (xi1<=xj2 and yi1<=yj2)) or \
               ((xj2>=xi1 and yj2>=yi1) and (xj2<=xi2 and yj2<=yi2)) or \
               ((xj1>=xi1 and yj1>=yi1) and (xj1<=xi2 and yj1<=yi2)) :                
                bboxs[j] = [min(xi1, xj1), min(yi1, yj1), max(xi2, xj2), max(yi2, yj2)] # 将bboxs[j]替换为大的整合框
                if i not in remove_list: # 将i记录在remove_list里面
                    remove_list.append(i)
    # 将remove_list内的元素删除
    sub_record = 0
    for i in remove_list:
        i -= sub_record
        bboxs.pop(i)
        sub_record += 1
    return bboxs

def create_mask_and_class_ids(labelmap_path):
    labelmap = skimage.io.imread(labelmap_path)
    masks, class_ids = divide_mask(labelmap)
    mask_output = []
    class_id_output = []

    for m in range(len(masks)):
        mask = masks[m]
        class_id = class_ids[m]

        binary = mask.copy().astype('uint8')*255 # 将mask转化为二值图
        _, contours, hierarchy = cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) # 二值图取contours

        # 从contours中提取bbox
        bboxs = []
        for contour in contours:
            x1, y1 = contour[0][0]
            x2, y2 = contour[0][0]
            for c in contour:
                c = c[0]
                x, y = c[0], c[1]
                if x<x1: x1=x
                if y<y1: y1=y
                if x>x2: x2=x
                if y>y2: y2=y
            bboxs.append([x1, y1, x2, y2])

        # 将bbox中互相包含的整合为一个，整合标准：只要有重叠，就整合为一个更大的bbox
        bboxs = combine_bboxs(bboxs)
        # 
        for bbox in bboxs:
            mask_bbox = np.zeros_like(mask)
            x1, y1, x2, y2 = bbox
            mask_bbox[y1:y2+1, x1:x2+1] = np.where(mask[y1:y2+1, x1:x2+1]==1, 1, 0)
            mask_output.append(mask_bbox)
            class_id_output.append(class_id)
    mask_output = np.array(mask_output).transpose(1,2,0)
    class_id_output = np.array(class_id_output, dtype=np.int32)
    return mask_output, class_id_output

#==============================================================================
# calculate mIOU
#==============================================================================
    
def create_labelmap(masks, class_ids, scores, config):
    # 将masks合并为一个labelmap
    # 按照scores降序排序
    idx_sort = np.argsort(scores)[::-1]
    class_ids = class_ids[idx_sort]
    scores = scores[idx_sort]
    masks = masks[..., idx_sort]

    image_shape = config.IMAGE_SHAPE
    labelmap = np.zeros((image_shape[0], image_shape[1]))
    for i in range(len(class_ids)-1,-1,-1):
        idx = np.where(masks[..., i] == True)
        labelmap[idx[0], idx[1]] = class_ids[i]
    labelmap = labelmap.astype('uint8')
    return labelmap

def cal_overlaps(labelmap_gt, labelmap_pred, config):
    # 计算mIOU: Jaccard Index = TP/(TP+FP+FN)
    # labelmap_gt, labelmap_pred：【1024，1024】，数值为0~3
    # 多图拼接的时候使用intersections，unions，单图使用overlaps
    intersections = []
    unions = []
    overlaps = []
    for class_target in range(1,config.NUM_CLASSES):
        # 如果class_target不存在于labelmap_pred和labelmap_pred_target中，则全部置零
        if (np.sum(labelmap_gt==class_target)+np.sum(labelmap_pred==class_target)) == 0:
            intersection = 0
            union = 0
            overlap = None
        # 生成labelmap_pred_target和labelmap_gt_target，他们在class_target位置的值为1，其余位置为0
        else:
            labelmap_pred_target = np.zeros_like(labelmap_pred)
            labelmap_pred_target[np.where(labelmap_pred==class_target)] = 1
            labelmap_gt_target = np.zeros_like(labelmap_gt)
            labelmap_gt_target[np.where(labelmap_gt==class_target)] = 1    
            intersection = (labelmap_pred_target*labelmap_gt_target).astype('bool')
            union = (labelmap_pred_target+labelmap_gt_target).astype('bool')
            overlap = np.sum(intersection)/np.sum(union)
        intersections.append(np.sum(intersection))
        unions.append(np.sum(union))
        overlaps.append(overlap)
    return intersections, unions, overlaps
    
if __name__ == 'main':
    labelmap_gt = np.zeros((16,16))
    labelmap_gt[:8,:8] = 1
    labelmap_gt[4:,12:] = 2
    labelmap_pred = np.zeros((16,16))
    labelmap_pred[1:9,1:9] = 1
    labelmap_pred[12:,12:] = 2
    _, _, overlaps = utils.cal_overlaps(labelmap_gt, labelmap_pred, config)
    print (overlaps)
    
#==============================================================================
# Evaluation
#==============================================================================
    
def concat_images(image_paths, boxes, image_concat_shape, mode, im_show=False):
    # image_paths: list of image_paths that to be concatenated
    # boxes: list of (x1, y1, x2, y2)
    # image_concat_shape: (w, h) of concatenated image
    # mode: 'RGB' or 'L'
    image_frame = Image.new(mode, image_concat_shape, "black")
    imageNum = len(image_paths)

    for i in range(imageNum):
        image = Image.open(image_paths[i])
        image_frame.paste(image, boxes[i])
    if im_show:
        w, h = image_frame.size
        image_show = image_frame.resize((w//10, h//10))
        image_show = np.array(image_show)
        fig = plt.figure(figsize=(16,16))
        if mode == 'L':
            plt.imshow(image_show, 'gray')
        else:
            plt.imshow(image_show)
        plt.show()
    return image_frame

def create_boxes(image_paths):
    # 按照'image'的位置加3位，锁定x1,y1,interval
    boxes = []
    for path in image_paths:
        _, end = re.search('image',path).span()
        x1, y1, interval = (int(x) for x in path[end+3:].split('.')[0].split('_')[:3])
        x2 = x1+interval
        y2 = y1+interval
        boxes.append((x1, y1, x2, y2))
    return boxes

class XYNYDataset(mrcnnUtils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_image_and_labelmap_paths(self, all_image_paths, with_white=False):
        """Generate image_info by image_paths and labelmap_paths.
        all_image_paths: list of image_path and labelmap_paths.
        """
        # Add classes
        self.add_class("XYNY", 1, "tabacoo")
        self.add_class("XYNY", 2, "corn")
        self.add_class("XYNY", 3, "coix")
        if with_white:
            self.add_class("XYNY", 4, "with_white")
        
        # Add images
        for i in range(len(all_image_paths)):
            image_path = all_image_paths[i][0]
            labelmap_path = all_image_paths[i][1]
            LMB_path = image_path.split('.')[0]+'_LMB.png'
            self.add_image("XYNY", image_id=i, path=image_path, labelmap_path=labelmap_path, LMB_path=LMB_path)
    
    def load_mask(self, image_id):
        # Load image
        labelmap_path = self.image_info[image_id]['labelmap_path']
        mask, class_ids = create_mask_and_class_ids(labelmap_path)
        return mask.astype('bool'), class_ids
    
    def load_LMB(self, image_id):
        LMB_path = self.image_info[image_id]['LMB_path']
        LMB = skimage.io.imread(LMB_path)
        return LMB
    
class XYNYTESTDataset(mrcnnUtils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def load_image_and_labelmap_paths(self, all_image_paths, with_white=False):
        """Generate image_info by image_paths and labelmap_paths.
        all_image_paths: list of image_path and labelmap_paths.
        """
        # Add classes
        self.add_class("XYNY", 1, "tabacoo")
        self.add_class("XYNY", 2, "corn")
        self.add_class("XYNY", 3, "coix")
        if with_white:
            self.add_class("XYNY", 4, "with_white")
        # Add images
        for i in range(len(all_image_paths)):
            self.add_image("XYNY", image_id=i, path=all_image_paths[i][0], LMB_path=all_image_paths[i][1])
        
    def load_LMB(self, image_id):
        # Load image
        LMB_path = self.image_info[image_id]['LMB_path']
        LMB = skimage.io.imread(LMB_path)
        return LMB
    
def generate_dataset(test_target, data_name):
    with open(data_name, 'r') as f:
        f_lines = f.readlines()
    image_paths = []
    LBM_paths = []
    for content in f_lines:
        if test_target in content:
            image_path, _, _, _, LBM_path = content.strip().split(',')
            image_paths.append(image_path)
            LBM_paths.append(LBM_path)
    all_image_paths = list(zip(image_paths, LBM_paths))

    dataset_test = XYNYTESTDataset()
    dataset_test.load_image_and_labelmap_paths(all_image_paths)
    dataset_test.prepare()
    return dataset_test

def get_image_shape(test_target):
    IMAGE_SIZE_DICT = {'image_1':(47161,50141), 'image_2':(77470,46050),
                       'image_3':(37241,19903), 'image_4':(25936,28832),}
    return IMAGE_SIZE_DICT[test_target]

def create_model_stamp(model_path):
    stamp1, stamp2 = model_path.split('.')[-2].split('\\')[-2:]
    model_stamp = stamp1[4:]+'_'+stamp2[-4:]
    return model_stamp
            
#==============================================================================
# config
#==============================================================================

class XYNYConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "XYNY"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # No resize because the image size is (1024,1024,3)
    IMAGE_RESIZE_MODE = "none"

class XYNYConfig_WW(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "XYNY"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 4  # background + 3 shapes

    # No resize because the image size is (1024,1024,3)
    IMAGE_RESIZE_MODE = "none"
    
class XYNYConfig_V2(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "XYNY"

    # Train on 1 GPU and 2 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 2 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 3  # background + 3 shapes

    # No resize because the image size is (1024,1024,3)
    IMAGE_RESIZE_MODE = "none"

    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    RPN_ANCHOR_STRIDE = 2
    
    POST_NMS_ROIS_INFERENCE = 2500
    POST_NMS_ROIS_TRAINING = 5000
    PRE_NMS_LIMIT = 15000 # 保持为POST_NMS_ROIS_TRAINING的3倍

def check_config(config):
    if 1024 in config.RPN_ANCHOR_SCALES:
        print ('config is V2')
    else:
        print ('config is V1')
    print ('DETECTION_MIN_CONFIDENCE :', config.DETECTION_MIN_CONFIDENCE)