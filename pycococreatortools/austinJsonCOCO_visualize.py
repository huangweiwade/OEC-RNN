#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
import cv2
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
#from google.colab.patches import cv2_imshow
import json

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
cfg = get_cfg()
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

if __name__ == "__main__":
    main_dir  = r'/home/super/桌面/polyrnn-pp-pytorch/Aerial_image'
    json_list = glob(os.path.join(main_dir,'jsonCOCO','austin*.json'))[1:2]
    #json_path = np.random.choice(json_list)
    
    dataset_dicts = []
    for idx, json_path in enumerate(json_list):
        with open(json_path,'r') as f:
            dictjson = json.load(f)
        record = {}
        
        filename = os.path.join(main_dir,'images',dictjson['images']['file_name'])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        objs = []
        for ann in dictjson["annotations"]: ## ann = np.random.choice(dictjson["annotations"])
            try:
                x,y,w,h = ann['bbox'][0],ann['bbox'][1],ann['bbox'][2],ann['bbox'][3]
                kp = [x,y,x+w,y+h]
                a = np.array(kp).reshape(-1,2)
                b = 2*np.ones((a.shape[0],1))
                c = [np.hstack((a,b)).flatten()]
                obj = {
                        "bbox": ann['bbox'],
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": ann['segmentation'],
                        "keypoints":c,
                        "category_id": 0,
                      }
            except:pass
            objs.append(obj)
        record["annotations"] = objs
        
        dataset_dicts.append(record)
        
#    json_filename = os.path.join(main_dir,'annotation5000.json')    
#    with open(json_filename, 'w') as jf:
#        json.dump(dataset_dicts, jf, indent=2)
    
    
    d = np.random.choice(dataset_dicts)
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], scale=1.0)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image())

