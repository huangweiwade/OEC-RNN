#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random
from tqdm import tqdm
import matplotlib.pyplot as plt
from glob import glob
# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import DatasetEvaluator,inference_on_dataset,print_csv_format

def austin_building(main_dir):
    main_dir = '/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/TrainForDetectron2'
    json_filename = glob(os.path.join(main_dir,'annotation.json'))[0]
    with open(json_filename,'r') as f:
        dataset_dicts = json.load(f)
    for d in dataset_dicts:
        for ann in d['annotations']: # ann=d['annotations'][0]["bbox_mode"]
            ann["bbox_mode"] = BoxMode.XYWH_ABS
    return dataset_dicts

if __name__ == "__main__":
    try:
        for d in ["TrainForDetectron2", "ValForDetectron2"]:
            DatasetCatalog.register('austin_' + d, 
                                    lambda d=d: austin_building('/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/' + d))
            MetadataCatalog.get('austin_' + d).set(thing_classes=["building"])
            dataset_dicts = DatasetCatalog.get("austin_TrainForDetectron2")
    except:
        pass
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("austin_TrainForDetectron2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
#    cfg.INPUT.MIN_SIZE_TRAIN = (800,1000)
    cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
#    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 100
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2   # 两类就对了

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
#    trainer = DefaultTrainer(cfg)
#    trainer.resume_or_load(resume=True)
#    trainer.train()
    
#######################################################################

    Dtrainer = DefaultTrainer(cfg)
    Dtrainer.resume_or_load(resume=False)  # dir(Dtrainer)
    Dtrainer.model.train()
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.engine.train_loop import SimpleTrainer
    from detectron2.evaluation import verify_results
    from detectron2.utils import comm
    STrainer = SimpleTrainer(Dtrainer.model,   # dir(STrainer)
                             Dtrainer.data_loader,
                             Dtrainer.optimizer)
    
    STrainer.scheduler = Dtrainer.build_lr_scheduler(cfg, Dtrainer.optimizer)
    STrainer.checkpointer = DetectionCheckpointer(Dtrainer.model,cfg.OUTPUT_DIR,
                                              optimizer=Dtrainer.optimizer,
                                              scheduler=STrainer.scheduler)
    STrainer.start_iter = 0
    STrainer.max_iter = cfg.SOLVER.MAX_ITER
    STrainer.cfg = cfg
    STrainer.register_hooks(Dtrainer.build_hooks())
    
    #(STrainer.train(Dtrainer.start_iter, Dtrainer.max_iter))
    import logging
    from detectron2.utils.events import EventStorage
    import time
    start_iter = STrainer.start_iter
    max_iter = STrainer.max_iter 
    logger = logging.getLogger("detectron2")
    logger.info("Starting training from iteration {}".format(start_iter))
    
    STrainer.model = Dtrainer.model
    STrainer.data_loader = Dtrainer.data_loader
    STrainer._data_loader_iter = iter(STrainer.data_loader)
    STrainer.optimizer = Dtrainer.optimizer
    
    with EventStorage(start_iter) as STrainer.storage:
        try:
            STrainer.before_train()
            for STrainer.iter in range(start_iter, max_iter):
                STrainer.before_step()
#                STrainer.run_step()
                
                start = time.perf_counter()
                data = next(STrainer._data_loader_iter)
                data_time = time.perf_counter() - start
                loss_dict = STrainer.model(data)
                losses = sum(loss_dict.values())
                STrainer.optimizer.zero_grad()
                losses.backward()
                STrainer._write_metrics(loss_dict, data_time)
                STrainer.optimizer.step()
                        
                STrainer.after_step()
            STrainer.iter += 1
        except Exception:
            logger.exception("Exception during training:")
            raise
        finally:
            STrainer.after_train()
    
    if len(Dtrainer.cfg.TEST.EXPECTED_RESULTS) and comm.is_main_process():
        verify_results(Dtrainer.cfg, Dtrainer.test(cfg, Dtrainer.model))
    
    
    
    










