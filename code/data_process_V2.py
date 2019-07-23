#!/usr/bin/env python
# coding: utf-8

# #### data process 根据比例进行缩放切割

# In[1]:


import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import sys
import PIL
from PIL import Image
import skimage.io
import re
import random
import utils
import zipfile

#%%
DATA_DIR = '..\\data'
#%%
# 解压图片
def unzip_data(zipfile_name, save_dir=None):
    # 解压zipfile_name的文件到save_dir，如果save_dir为None，则解压到zipfile_name的同一根目录下
    if save_dir is None:
        save_dir = '\\'.join(zipfile_name.split('\\')[:-1])
    with zipfile.ZipFile(zipfile_name, 'r') as zipf:
        zipf.extractall(save_dir)

zipfile_dir = DATA_DIR
zipfile_name_list = [
                     'jingwei_round1_submit_20190619.zip',
                     'jingwei_round1_train_20190619.zip',
                     'jingwei_round1_test_a_20190619.zip',
                     ]
for zipfile_name in zipfile_name_list:
    zipfile_name_total = os.path.join(zipfile_dir, zipfile_name)
    try:
        unzip_data(zipfile_name_total)
        print (zipfile_name_total, 'has been unzipped successfully.')
    except:
        print (zipfile_name_total, 'Error, please check the zip files')

#%%

class image_divide_processor():
    def __init__(self, image_path_list, labelmap_path_list=None, save_dir=None, interval=1024, no_zero=True, small_ratio=1, pad=0):
        # image_path: list of image_path
        # labelmap_path: list of labelmap_path
        # interval: image divide interval
        # no_zero: no_zero为True时，全部为zero的图片不进行保存
        # small_ratio: 图片经过small_ratio的缩小倍数后进行处理
        # pad: 在图片的上方和左方pad 0
        if (labelmap_path_list is None) or (len(labelmap_path_list) == 0):
            self.is_test = True
        else:
            assert len(image_path_list)==len(labelmap_path_list)
            self.is_test = False
        self.image_path_list = image_path_list
        self.labelmap_path_list = labelmap_path_list
        self.save_dir = save_dir
        self.interval = interval
        self.no_zero = no_zero
        self.small_ratio = small_ratio
        self.pad = pad
        
    def divide_image(self):       
        self.records = []
        if not self.is_test: # train data mode
            for i, image_path in enumerate(self.image_path_list):
                image_record = self.crop_and_save_image(image_path)
                labelmap_path = self.labelmap_path_list[i]
                record = self.crop_and_save_labelmap(labelmap_path, image_record)
                self.records.extend(record)            
        if self.is_test: #test data mode
            for i, image_path in enumerate(self.image_path_list):
                record = self.crop_and_save_test_image(image_path)
                self.records.extend(record)
        print ('Function: <divide_image> has been finished. The number of records is ', len(self.records))
        
    def write_records(self, save_path):
        records_to_save = [record+'\n' for record in self.records]
        #save_path = os.path.join(EXTRACT_DIR, 'train_data.txt')
        with open(save_path, 'w+') as f:
            f.writelines(records_to_save)
    
    def open_pad_resize_image(self, image_path):
        PIL.Image.MAX_IMAGE_PIXELS = None        
        image = PIL.Image.open(image_path)
        pad = self.pad
        width,height = image.size
        # 确认输入是image/'RGBA'还是labelmap/'L'
        if image.mode=='RGBA':
            image = image.convert('RGB')
            resize_mode = Image.BILINEAR # 输入为image，则resize mode使用双线性
        elif image.mode=='L':
            resize_mode = Image.NEAREST # 输入为labelmap，则resize mode使用最近值
        else:
            print ('Warning: image mode is neither RGBA nor L, it is: ', image.mode)
            resize_mode = Image.BILINEAR
        # 如果pad大于0，则在图片的上方与左侧增加pad个0
        if pad>0:            
            image_frame = Image.new(image.mode, (width+pad, height+pad), "black")
            box = (pad, pad, width+pad, height+pad)
            image_frame.paste(image, box)
            image.close()
            image = image_frame
        # 如果small_ratio不为1，则按比例缩小图片
        if self.small_ratio != 1:
            width,height = image.size
            width,height = width//self.small_ratio, height//self.small_ratio
            image = image.resize((width, height), resize_mode)
        return image
        
    def crop_and_save_image(self, image_path):
        # 输入image_path，将image按照interval进行分割，保存到save_dir中
        # no_zero为True时，全部为zero的图片不进行保存
        # 最后输出一个record_text的list，每个元素为（image_path，box_x, box_y, interval）
        interval = self.interval
        image = self.open_pad_resize_image(image_path)
        width,height = image.size
        ws, hs = width//interval, height//interval
        record_text = []
        for w in range(ws)[:]:
            for h in range(hs)[:]:
                step = w*hs+h+1
                print (step,'/',ws*hs)
                box = [w*interval,h*interval,(w+1)*interval,(h+1)*interval]
                image_crop = image.crop(box)
                image_crop = np.array(image_crop)
                if self.no_zero and np.sum(image_crop)==0:
                    print ('image_crop is zero, pass', box)
                    continue
                file_name = image_path.split('\\')[-1].split('.')[0]
                if self.small_ratio == 1:
                    # i.e: image_1_0_37888_1024.png
                    file_name = file_name+'_'+str(box[0])+'_'+str(box[1])+'_'+str(interval)+'.png'
                else:
                    # i.e: image_1_0_37888_1024_10small.png
                    file_name = file_name+'_'+str(box[0])+'_'+str(box[1])+'_'+str(interval)+'_'+str(self.small_ratio)+'small'+'.png'
                file_all_name = os.path.join(self.save_dir, file_name)
                skimage.io.imsave(file_all_name, image_crop)
                record = ','.join([file_all_name,str(box[0]),str(box[1]),str(interval)])
                record_text.append(record)
                print ('%s is created successfully' %file_all_name, box)
        image.close()
        return record_text

    def crop_and_save_labelmap(self, labelmap_path, record_text):
        # 输入labelmap和record_text，根据record_text的信息生成labelmap
        labelmap = self.open_pad_resize_image(labelmap_path)
        record_result = []
        for i, record in enumerate(record_text):
            print (i+1,'/',len(record_text))
            image_file_name,x,y,interval = record.split(',')
            # crop labelmap
            box = [int(x),int(y),int(x)+int(interval),int(y)+int(interval)]
            labelmap_crop = labelmap.crop(box)
            labelmap_crop = np.array(labelmap_crop)
            # generate file_name
            start,end = re.search('.png', image_file_name).span()
            labelmap_file_name = image_file_name[:start]+'_label'+image_file_name[start:]
            record = record+','+labelmap_file_name
            skimage.io.imsave(labelmap_file_name, labelmap_crop)
            record_result.append(record)
        labelmap.close()
        return record_result
    
    def generate_labelmap_base(self,image_crop):
        # 根据image_crop生成labelmap_base。
        # labelmap_base与image_crop同高同宽，image_crop为（0，0，0）的地方置为0，其余为1.
        # 意在之后将labelmap_base与labelmap_predict相乘，原(0,0,0)的地方直接作为背景
        labelmap_base = np.logical_not(np.prod(image_crop == 0, axis=2))
        return labelmap_base.astype('uint8')

    def crop_and_save_test_image(self, image_path):
        # 输入image_path，将image按照interval进行分割，保存到save_dir中
        # no_zero为True时，全部为zero的图片不进行保存
        # 最后输出一个record_text的list，每个元素为（image_path，box_x,box_y,interval）
        interval = self.interval
        image = self.open_pad_resize_image(image_path)
        width,height = image.size
        ws, hs = width//interval, height//interval
        record_text = []
        for w in range(ws):
            for h in range(hs):
                step = w*hs+h+1
                print (step,'/',ws*hs)
                box = [w*interval,h*interval,(w+1)*interval,(h+1)*interval]
                image_crop = image.crop(box)
                image_crop = np.array(image_crop)
                if self.no_zero and np.sum(image_crop)==0:
                    print ('image_crop is zero, pass', box)
                    continue
                # 保存image
                image_name = image_path.split('\\')[-1].split('.')[0]
                if self.small_ratio == 1:
                    # i.e: image_1_0_37888_1024.png
                    imagefile_name = image_name+'_'+str(box[0])+'_'+str(box[1])+'_'+str(interval)+'.png'
                else:
                    # i.e: image_1_0_37888_1024_10small.png
                    imagefile_name = image_name+'_'+str(box[0])+'_'+str(box[1])+'_'+str(interval)+'_'+str(self.small_ratio)+'small'+'.png'
                imagefile_name = os.path.join(self.save_dir, imagefile_name)
                skimage.io.imsave(imagefile_name, image_crop)
                # 保存labelmap_base
                labelmap_base = self.generate_labelmap_base(image_crop)
                start,end = re.search('.png', imagefile_name).span()
                labelmap_base_name = imagefile_name[:start]+'_LMB'+imagefile_name[start:]
                skimage.io.imsave(labelmap_base_name, labelmap_base)

                record = ','.join([imagefile_name,str(box[0]),str(box[1]),str(interval),labelmap_base_name])            
                record_text.append(record)
                print ('%s is created successfully' %imagefile_name, box)
        image.close()
        return record_text

#%%
# 切分图片image_1、image_2，制作train_data。按照ratio=1， pad=0,256,512,768的参数进行切分

SMALL_RATIO = 1
PAD_list = [0, 256, 512, 768]
EXTRACT_DIR = os.path.join(DATA_DIR, 'extract_data')
if not os.path.exists(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)
    
for PAD in PAD_list:
    TRAIN_DIR_NAME = 'train_data_%spad'%PAD
    TRAIN_DIR = os.path.join(EXTRACT_DIR, TRAIN_DIR_NAME)
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    
    OFFICAL_TRAIN_DIR = os.path.join(DATA_DIR, 'jingwei_round1_train_20190619')
    OFFICAL_TEST_DIR = os.path.join(DATA_DIR, 'jingwei_round1_test_a_20190619')
    
    image_path_list = []
    for img in ['image_1.png', 'image_2.png']:
        image_path_list.append(os.path.join(OFFICAL_TRAIN_DIR, img))
    labelmap_path_list = []
    for img in ['image_1_label.png', 'image_2_label.png']:
        labelmap_path_list.append(os.path.join(OFFICAL_TRAIN_DIR, img))
    save_dir = TRAIN_DIR
    image_divide_fn = image_divide_processor(image_path_list, labelmap_path_list, save_dir, small_ratio=SMALL_RATIO, pad=PAD)
    image_divide_fn.divide_image()
    
    save_path = os.path.join(EXTRACT_DIR, '%s.txt' %TRAIN_DIR_NAME)
    image_divide_fn.write_records(save_path)
    
    # 制作LMB
    image_divide_fn.is_test = True
    image_divide_fn.divide_image()
    save_path = os.path.join(EXTRACT_DIR, '%s_LMB.txt' %TRAIN_DIR_NAME)
    image_divide_fn.write_records(save_path)

#%%
# 遍历图片，剔除仅含有BG的图片，将含有class的图片列表提取并保存
PAD_list = [0, 256, 512, 768]
for PAD in PAD_list:
    TRAIN_DIR_NAME = 'train_data_%spad'%PAD
    txt_path = os.path.join(EXTRACT_DIR, '%s.txt'%TRAIN_DIR_NAME)
    with open(txt_path, 'r+') as f:
        f_lines = f.readlines()
    
    f_lines_has_class = []
    for line in f_lines:
        _, _, _, _, labelmap_path = line.strip().split(',')
        labelmap = skimage.io.imread(labelmap_path)
        if np.sum(labelmap)!=0:
            f_lines_has_class.append(line)
            
    save_path = os.path.join(EXTRACT_DIR, '%s_has_class.txt'%TRAIN_DIR_NAME)
    with open(save_path, 'w+') as f:
        f.writelines(f_lines_has_class)
        
#%%
# 切分图片image_3、image_4，制作test_data。按照ratio=1， pad=0,341,682的参数进行切分
SMALL_RATIO = 1
PAD_list = [0, 341, 682]
EXTRACT_DIR = os.path.join(DATA_DIR, 'extract_data')
if not os.path.exists(EXTRACT_DIR):
    os.mkdir(EXTRACT_DIR)

for PAD in PAD_list:
    TEST_DIR_NAME = 'test_data_%spad'%PAD
    TEST_DIR = os.path.join(EXTRACT_DIR, TEST_DIR_NAME)
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)

    image_path_list = []
    for img in ['image_3.png', 'image_4.png']:
        image_path_list.append(os.path.join(OFFICAL_TEST_DIR, img))
    labelmap_path_list = []

    image_divide_fn = image_divide_processor(image_path_list, labelmap_path_list, save_dir=TEST_DIR, small_ratio=SMALL_RATIO, pad=PAD)
    image_divide_fn.divide_image()
    
    save_path = os.path.join(EXTRACT_DIR, '%s.txt'%TEST_DIR_NAME)
    image_divide_fn.write_records(save_path)
