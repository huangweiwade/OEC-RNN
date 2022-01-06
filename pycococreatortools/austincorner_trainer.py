#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import matplotlib.pyplot as plt
from glob import glob
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

#cfg = get_cfg()
#cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#cfg.DATASETS.TRAIN = ("balloon_train",)
#cfg.DATASETS.TEST = ()
#cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.SOLVER.IMS_PER_BATCH = 2
#cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
#cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
#
#os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#trainer = DefaultTrainer(cfg) 
#trainer.resume_or_load(resume=False)
#trainer.train()

def austin_building(main_dir):
#    main_dir = '/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/TrainForDetectron2'
    json_filename = glob(os.path.join(main_dir,'annotation.json'))[0]
    with open(json_filename,'r') as f:
        dataset_dicts = json.load(f)
  
    for d in dataset_dicts:
        for ann in d['annotations']: # ann=d['annotations'][0]["bbox_mode"]
            ann["bbox_mode"] = BoxMode.XYWH_ABS
            
    return dataset_dicts

#    d = np.random.choice(dataset_dicts)
#    img = cv2.imread(d["file_name"])
#    visualizer = Visualizer(img[:, :, ::-1], scale=1.0)
#    out = visualizer.draw_dataset_dict(d)
#    plt.imshow(out.get_image())

def Train():
    try:
        for d in ["TrainForDetectron2", "ValForDetectron2"]:
            DatasetCatalog.register('austinc_' + d, 
                                    lambda d=d: austin_building('/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/' + d))
            MetadataCatalog.get('austinc_' + d).set(thing_classes=["building"])
            MetadataCatalog.get('austinc_' + d).set(keypoint_names=['corner'])
#            MetadataCatalog.get('austinc_' + d).set(keypoint_names=['corner'+str(i) for i in range(20)])
            MetadataCatalog.get('austinc_' + d).set(keypoint_flip_map=[])
#            MetadataCatalog.get('austinc_' + d).set(keypoint_flip_map=[('corner'+str(i) ,'corner'+str(i)) for i in range(20)])
        dataset_dicts = DatasetCatalog.get("austinc_ValForDetectron2")
    except:
        pass
    
    cfg = get_cfg()
    cfg.merge_from_file(cfgfile)
    cfg.INPUT.RANDOM_FLIP = "none"
#    cfg.INPUT.MIN_SIZE_TRAIN = (800,1000)

    cfg.DATASETS.TRAIN = ("austinc_ValForDetectron2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
#    cfg.MODEL.WEIGHTS = "/home/super/桌面/detectron2/model_final_f10217.pkl"
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = "model_final_a6e10b.pkl"

    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 200
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 20
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 3
#    cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = False

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()


def Predict():
    dataset_dicts = DatasetCatalog.get("austinc_ValForDetectron2")

    d = np.random.choice(dataset_dicts)
    im = cv2.imread(d["file_name"])

    cfg = get_cfg()
    cfg.merge_from_file(cfgfile)
    cfg.DATASETS.TEST = ("austinc_ValForDetectron2", )
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.WEIGHTS = "model_final_a6e10b.pkl"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],  scale=1)
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(v.get_image())

if __name__ == "__main__":
    cfgfile ='detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
    Train()
#    Predict()
