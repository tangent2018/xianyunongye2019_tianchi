#!/usr/bin/env python
# coding: utf-8

import os
import sys
import random
import re
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import skimage
import zipfile

from mrcnn.config import Config
from mrcnn import utils as mrcnnUtils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

import utils

#%%
# config

config = utils.XYNYConfig()
config.display()

#%%
def create_score_map(test_target, data_name, config, ratio, model_path=None, save_score_map=False, save_image=False):
    MODEL_DIR = '.\\logs'
    # check config
    utils.check_config(config)
    # create model
    model_inference = modellib.MaskRCNN(mode="inference", config=config, model_dir=MODEL_DIR)
    if model_path is None:
        model_path = model_inference.find_last()
    print("Loading weights from ", model_path)    
    model_inference.load_weights(model_path, by_name=True)
    # generate dataset
    # 检查data是否包含small
    if 'pad' in data_name:
        _start, _end = re.search('pad', data_name).span()
        data_pad = int(data_name[:_start].split('_')[-1])
    else:
        data_pad = 0
    print ('test_target: %s, data_pad: %s' %(test_target, data_pad))
    dataset_test = utils.generate_dataset(test_target, data_name)
    image_shape = utils.get_image_shape(test_target)
    idsNum = len(dataset_test.image_ids)

    # generate score_map_total
    score_map_total = np.zeros((image_shape[1]//ratio, image_shape[0]//ratio, config.NUM_CLASSES-1), dtype=np.float32)

    for image_id in dataset_test.image_ids[:]:
        if image_id%50==0:
            print ('%s/%s' %(image_id,idsNum))
        # generate image
        image_path = dataset_test.image_info[image_id]['path']
        image = skimage.io.imread(image_path)
        # generate LMB
        LMB_path = dataset_test.image_info[image_id]['LMB_path']
        LMB = skimage.io.imread(LMB_path)
        # generate x1,y1,interval
        x1, y1, interval = image_path.split('\\')[-1].split('.')[-2].split('_')[2:5]
        def _ratio(x):
            return int(x)//ratio
        x1_ratio, y1_ratio, interval_ratio = list(map(_ratio, [x1, y1, interval]))
        x1_ratio -= data_pad//ratio
        y1_ratio -= data_pad//ratio
        score_map_total_x1, score_map_total_y1 = x1_ratio, y1_ratio
        score_map_total_x2 = score_map_total_x1+interval_ratio
        score_map_total_y2 = score_map_total_y1+interval_ratio
        score_map_x1, score_map_y1 = 0, 0        
        if x1_ratio<0:
            score_map_total_x1 = 0
            score_map_x1 = -x1_ratio
        if y1_ratio<0:
            score_map_total_y1 = 0
            score_map_y1 = -y1_ratio
            
        # predict results    
        results = model_inference.detect([image], verbose=0)
        r = results[0]        
        
        for n,class_id in enumerate(r['class_ids']):
            # generate mask_n
            mask_n = r['masks'][:,:,n].astype(np.float32)
            mask_n = mask_n*LMB
            if ratio != 1:
                mask_n = cv2.resize(mask_n, (interval_ratio, interval_ratio), interpolation=cv2.INTER_NEAREST)
            # generate score_map
            score_n = r['scores'][n]
            score_map = mask_n*score_n
            # 取score_map_total和score_map对应pixel的较大值赋值到score_map_total
            score_map_total[score_map_total_y1:score_map_total_y2,score_map_total_x1:score_map_total_x2,class_id-1] =             np.maximum(score_map[score_map_y1:, score_map_x1:], 
                       score_map_total[score_map_total_y1:score_map_total_y2,score_map_total_x1:score_map_total_x2,class_id-1])
            
    # save phase        
    save_dir = '.\\merge_submit'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)    
    model_stamp = utils.create_model_stamp(model_path)
    image_name = '%s_%ssmall_%spad' %(test_target, ratio, data_pad)
    # save score_map_total as npy
    if save_score_map:
        npy_save_path = os.path.join(save_dir, '%s_scoremap_%s.npy' %(image_name, model_stamp))
        np.save(npy_save_path, score_map_total)
        print (npy_save_path, 'has been saved succefully.')
    # save and show score_map as png by class
    if save_image:       
        rows = config.NUM_CLASSES-1
        fig, axs = plt.subplots(rows, 1, figsize=(16,16*rows))
        
        if ratio==1:
            image_path = ".\\offical_data\\jingwei_round1_test_a_20190619\\%s.png" %test_target
        else:
            image_path = ".\\overview\\%s_%ssmall.png" %(test_target, ratio)
        Image.MAX_IMAGE_PIXELS = None
        image_frame_oringin = skimage.io.imread(image_path)
        
        for class_id in range(1, config.NUM_CLASSES):
            score_map = score_map_total[:,:,class_id-1]
            image_frame = image_frame_oringin.copy()
            color = utils.COLOR_MAP[class_id]
            for c in range(3):
                image_frame[:, :, c] = image_frame[:, :, c] *(1 - score_map) + score_map * color[c]

            axs[class_id-1].imshow(image_frame)
            image_save_path = os.path.join(save_dir, '%s_class%d_%s.png' %(image_name, class_id, model_stamp))
            skimage.io.imsave(image_save_path, image_frame)
            print (image_save_path, 'has been saved succefully.')
            
    return score_map_total, model_path

#%%
# generate score_map

config.DETECTION_MIN_CONFIDENCE = 0.4
ratio = 3
test_target_list = ['image_3', 'image_4']
data_name_list = [
    '..\\data\\extract_data\\test_data_0pad.txt',
    '..\\data\\extract_data\\test_data_341pad.txt',
    '..\\data\\extract_data\\test_data_682pad.txt'
]
for test_target in test_target_list:
    for data_name in data_name_list:
        score_map_total, model_path = create_score_map(test_target, data_name, config, ratio=ratio, save_score_map=True)

#%%
def create_label_map(test_target, config, score_map_path_list, confidence_thre=0.7, ratio = 10, with_white=False):
    image_shape = utils.get_image_shape(test_target)
    w, h = image_shape[0]//ratio, image_shape[1]//ratio
    
    label_score_map_list = []
    for score_map_path in score_map_path_list:    
        score_map = np.load(score_map_path)
        # confidence_thre以下的置为0
        if not with_white:
            score_map = score_map[:,:,:3]
        score_map = np.where(score_map<confidence_thre, 0, score_map)
        # 计算label_score_map, [h, w, (class_id, score)]
        label_score_map = np.zeros((h,w,2))
        score_map_mask = np.sum(score_map, axis=2).astype(np.bool)
        label_score_map[:, :, 0] = (np.argmax(score_map, axis=2)+1)*score_map_mask
        label_score_map[:, :, 1] = (np.max(score_map, axis=2))
        label_score_map_list.append(label_score_map)

    label_map = label_score_map_list[0][:,:,0].copy()
    score_map = label_score_map_list[0][:,:,1].copy()
    for i, label_score_map in enumerate(label_score_map_list[1:]):
        label_map_next = label_score_map[:,:,0]
        score_map_next = label_score_map[:,:,1]
        # update label_map
        label_map = np.where(score_map>score_map_next, label_map, label_map_next)
        # update score_map
        score_map = np.where(score_map>score_map_next, score_map, score_map_next)
    label_map = label_map.astype(np.uint8)
    return label_map

#%%
# merge score_map
ratio = 3
merge_dir = '.\\merge_submit'
file_names = os.listdir(merge_dir)
model_stamp = utils.create_model_stamp(model_path)

for test_target in test_target_list:
    score_map_path_list = []
    for file_name in file_names:
        if test_target in file_name and model_stamp in file_name and 'npy' in file_name:
            score_map_path_list.append(os.path.join(merge_dir, file_name))
    print ('score_map_path_list: ', score_map_path_list)        
            
    confidence_thre=0.7
    conf_str = str(confidence_thre).split('.')[-1]
    label_map = create_label_map(test_target, config, score_map_path_list, confidence_thre=confidence_thre, ratio=ratio)
    
    save_dir = os.path.join(merge_dir, 'merge')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    image_name = '%s_%ssmall'%(test_target, ratio)
    
    label_map_save_path = os.path.join(save_dir, '%s_%sconf_labelmap_%s.png' %(image_name, conf_str, model_stamp))
    skimage.io.imsave(label_map_save_path, label_map)
    print (label_map_save_path, 'has been saved successfully.')        

    # resize label_map and save in '..\\submit'
    image_shape = utils.get_image_shape(test_target)
    label_map = cv2.resize(label_map, image_shape, interpolation=cv2.INTER_NEAREST)
    submit_dir = '..\\submit\\submit'
    if not os.path.exists(submit_dir):
        os.mkdir(submit_dir)
    submit_name = os.path.join(submit_dir, (test_target+'_predict.png'))
    skimage.io.imsave(submit_name, label_map)
    print (submit_name, 'has been saved successfully.')

#%%
# 文件压缩为zip
time_stamp = time.strftime('%Y%m%d_%H%M')
zipfilename = submit_dir+time_stamp+'.zip'
filelist = os.listdir(submit_dir)
filelist = [os.path.join(submit_dir, file) for file in filelist]
 
zf = zipfile.ZipFile(zipfilename, "w", zipfile.zlib.DEFLATED)
for tar in filelist:
    arcname = '.\\submit'+tar[len(submit_dir):]
    print (arcname)
    zf.write(tar, arcname)
zf.close()

 