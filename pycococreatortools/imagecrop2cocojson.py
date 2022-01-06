#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle,Circle
from matplotlib.collections import PatchCollection
from skimage import io
from scipy import ndimage as ndi
import sys
sys.path.append('/home/super/桌面/detectron2')
from pycococreatortools import pycococreatortools
# from pycocotools.coco import COCO
import json
from tqdm import tqdm

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

def CropInstanceToCoco(out_dir,image_paths,label_paths,image_shape):
    json_filename = os.path.join(out_dir,'annotation.json')
    outimg_dir = os.path.join(out_dir,'images')
    outgt_dir =  os.path.join(out_dir,'gts')
    out_instance = os.path.join(out_dir,'instances')
    
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)
    if not os.path.exists(outgt_dir):
        os.makedirs(outgt_dir)
    if not os.path.exists(out_instance):
        os.makedirs(out_instance)
        
       
    ## *** Clip the img to x00*x00 ***  
    batch_size = 1
    for batch_i in range(0, len(image_paths), batch_size):
        for image_file, gt_file in zip(image_paths[batch_i:batch_i+batch_size], label_paths[batch_i:batch_i+batch_size]):
            #image_file,gt_file = image_paths[0],label_paths[0]
            ###crop image_file
            image = cv2.imread(image_file)
            row = image.shape[0]
            col = image.shape[1]
            for i in range(int(row/image_shape[0])):
                for j in range(int(col/image_shape[1])):
                    subimage = image[i*image_shape[0]:(i+1)*image_shape[0],j*image_shape[1]:(j+1)*image_shape[1],:]
                    cv2.imwrite(outimg_dir+'/'+os.path.basename(image_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.tif',subimage)      
            i=int(row/image_shape[0])
            for j in range(int(col/image_shape[1])):
                subimage = image[row-image_shape[0]:row,j*image_shape[1]:(j+1)*image_shape[1],:]
                cv2.imwrite(outimg_dir+'/'+os.path.basename(image_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.tif',subimage)
            j=int(col/image_shape[0])
            for i in range(int(row/image_shape[0])):
                subimage = image[i*image_shape[0]:(i+1)*image_shape[0],col-image_shape[1]:col,:]      
                cv2.imwrite(outimg_dir+'/'+os.path.basename(image_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.tif',subimage)
            i,j=int(row/image_shape[0])+1,int(col/image_shape[0])+1
            subimage = image[row-image_shape[0]:row,col-image_shape[1]:col,:]
            cv2.imwrite(outimg_dir+'/'+os.path.basename(image_file).split('.')[0]+'_'+'%02d'%(i)+'_'+'%02d'%(j)+'.tif',subimage)
            
            ###crop gt_file
            image = io.imread(gt_file)
            row = image.shape[0]
            col = image.shape[1]
            for i in range(int(row/image_shape[0])):
                for j in range(int(col/image_shape[1])):
                    subimage = image[i*image_shape[0]:(i+1)*image_shape[0],j*image_shape[1]:(j+1)*image_shape[1]]
                    cv2.imwrite(outgt_dir+'/'+os.path.basename(gt_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.png',subimage)      
            i=int(row/image_shape[0])
            for j in range(int(col/image_shape[1])):
                subimage = image[row-image_shape[0]:row,j*image_shape[1]:(j+1)*image_shape[1]]
                cv2.imwrite(outgt_dir+'/'+os.path.basename(gt_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.png',subimage)
            j=int(col/image_shape[0])
            for i in range(int(row/image_shape[0])):
                subimage = image[i*image_shape[0]:(i+1)*image_shape[0],col-image_shape[1]:col]      
                cv2.imwrite(outgt_dir+'/'+os.path.basename(gt_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.png',subimage)
            i,j=int(row/image_shape[0])+1,int(col/image_shape[0])+1
            subimage = image[row-image_shape[0]:row,col-image_shape[1]:col]
            io.imsave(outgt_dir+'/'+os.path.basename(gt_file).split('.')[0]+'_'+'%02d'%(i)+'_'+'%02d'%(j)+'.png',subimage)
            
            
    ## *** This instance generator ***
    binaryPath = glob(os.path.join(outgt_dir,'*.png'))
    for binary_file in binaryPath:
        fn = os.path.basename(binary_file).split('.')[0]
        binary_gt  = io.imread(binary_file)
        instance_gt,n = ndi.label(binary_gt) #返回 label, num_features  plt.imshow(instance_gt)
        io.imsave(out_instance+'/'+ fn +'.tif',instance_gt)
    #randmap = np.vstack(([0,0,0],np.random.rand(n,3))) #因为0为黑色背景
    #plt.imshow(instance_gt, cmap=ListedColormap(randmap))
    
   
       
    
    ## *** Translate the img to coco style ***
    img_files = glob(outimg_dir + '//*.tif')
    ann_files = glob(out_instance+'//*.tif')

    image_id = 0
    ann_id = 0
    dataset_dicts =[]
    # go through each image and associated annotation
    for img_name, ann_name in tqdm(zip(img_files, ann_files)):# img_name = img_files[0]
        CATEGORIES = [{'id': 100,'name': 'building',}]
        dictjson = {"categories": CATEGORIES,"images": [],"annotations": []}
        
        image = Image.open(img_name)
        image_info = pycococreatortools.create_image_info(image_id, os.path.basename(img_name), image.size)
        dictjson["images"].append(image_info)
    
        annotation = io.imread(ann_name)
        for n_instn in range(1,np.max(annotation)+1):
            class_id = 100
            category_info = {'id': class_id, 'is_crowd': 0 }
            binary_mask = np.array(annotation==n_instn,dtype='uint8')
            annotation_info = pycococreatortools.create_annotation_info(ann_id, image_id, 
                                                                        category_info, binary_mask, image.size, 
                                                                        tolerance=1)
            if annotation_info is not None:
                dictjson["annotations"].append(annotation_info)
        image_id = image_id + 1
        ann_id =ann_id + 1
        
        ## *** Transform the detectron2 coco style ***
        filename = img_name
        height, width = cv2.imread(filename).shape[:2]
        
        record = {}
        record["file_name"] = filename
        record["image_id"] = image_id
        record["height"] = height
        record["width"] = width
      
        objs = []
        for ann in dictjson["annotations"]: ## ann = np.random.choice(dictjson["annotations"])
            try:
                obj = {
                        "bbox": ann['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": ann['segmentation'],
                        "category_id": 1,
                      }
            except:pass
            objs.append(obj)
        record["annotations"] = objs
        
        dataset_dicts.append(record)
        
    with open(json_filename, 'w') as jf:
        json.dump(dataset_dicts, jf, indent=2)

def visualjson(out_dir):
    json_filename = os.path.join(out_dir,'annotation.json')
    with open(json_filename,'r') as f:
        dataset_dicts = json.load(f)

    d = np.random.choice(dataset_dicts)
    for ann in d['annotations']: # ann=d['annotations'][0]["bbox_mode"]
        ann["bbox_mode"] = BoxMode.XYWH_ABS
    
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image())

if __name__ == "__main__":
    image_shape = [800,800]
    main_dir = r'/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/Train_forD2Yushu'
    image_paths = glob(os.path.join(main_dir,  '*.tif'))
    label_paths = glob(os.path.join(main_dir,  '*.png'))
    val_path    = glob(os.path.join(main_dir,  'Hangpai_After1.jpg'))
    
    out_dir = os.path.join(main_dir,'TrainForDetectron2')
    CropInstanceToCoco(out_dir,image_paths,label_paths,image_shape)
    
#    out_dir = os.path.join(main_dir,'ValForDetectron2')
#    CropInstanceToCoco(out_dir,val_path,label_paths,image_shape)  
    
    visualjson(out_dir)


