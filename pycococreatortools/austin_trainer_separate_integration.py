import detectron2
import sys
#sys.path.append(r'C:\Users\huang\detectron2\detectron2')
from detectron2.utils.logger import setup_logger

setup_logger()
import numpy as np
import torch
import os, json #, cv2, random
from tqdm import tqdm
from glob import glob
import os
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = "1" 

# import some common detectron2 utilities
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.structures import BoxMode
from detectron2.structures import Boxes
from detectron2.utils import comm
# from detectron2.evaluation import DatasetEvaluator,inference_on_dataset,print_csv_format

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
    Data_dir = r'/home/super/桌面/polyrnn-pp-pytorch/Aerial_image/'
    try:
        for d in ["TrainForDetectron2", "ValForDetectron2"]:
            DatasetCatalog.register('austin_' + d, 
                                    lambda d=d: austin_building(Data_dir + d))
            MetadataCatalog.get('austin_' + d).set(thing_classes=["building"])
            # MetadataCatalog.get('austin_' + d).set(keypoint_names=['corner'])
            dataset_dicts = DatasetCatalog.get("austin_TrainForDetectron2")
    except:
        pass
    
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/corner_rcnn_R_50_FPN_3x.yaml")
    # cfg.merge_from_file("configs\COCO-Keypoints\keypoint_rcnn_R_50_FPN_3x.yaml") 
    cfg.DATASETS.TRAIN = ("austin_TrainForDetectron2",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2

    cfg.MODEL.WEIGHTS = "output/model_final.pth"
#    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_0039999.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.REFERENCE_WORLD_SIZE = 0
    cfg.SOLVER.BASE_LR = 0.02
    cfg.SOLVER.MAX_ITER = 20000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2   # 两类就对了
    cfg.VIS_PERIOD = 0
    
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
    from detectron2.utils.events import EventStorage,get_event_storage
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
                    #  data0 = data[0]
                    #  image = data0['image'].numpy()
                    #  instance = data0['instances']
                    #  Boxes = instance.gt_boxes             # dir(Boxes)
                    #    bbox = Boxes[0].tensor.numpy()
                    #  PolygonMasks = instance.gt_masks      # dir(PolygonMasks)
                    #    mask = PolygonMasks[0].polygons
                data_time = time.perf_counter() - start
#                loss_dict = STrainer.model(data)
                
                batched_inputs = data
                images = STrainer.model.preprocess_image(batched_inputs)  # list
                    #  img0 = images[0].to("cpu").numpy()
                    #  img1 = images[1].to("cpu").numpy()
                gt_instances = [x["instances"].to(STrainer.model.device) for x in batched_inputs]
                    #  gt_instances[0].gt_boxes[0].tensor.to("cpu").numpy()
                    #  gt_instances[1].gt_masks[0].polygons
                features = STrainer.model.backbone(images.tensor)
                    #  dict,size=5, {'p2','p3','p4','p5','p6'}
                    #  p2 = features['p2'].to("cpu").detach().numpy()
                    #  p6 = features['p6'].to("cpu").detach().numpy()
                proposals, proposal_losses = STrainer.model.proposal_generator(images, features, gt_instances)
                    #  proposals: [Instance,Instance] size = 1000
                    #  p0 = proposals[0]               
                    #  pboxes   = p0.proposal_boxes.tensor.to("cpu").numpy()
                    #  objlogit = p0.objectness_logits                   
#                _, detector_losses = STrainer.model.roi_heads(images, features, proposals, gt_instances)
                
                targets = gt_instances
                if STrainer.model.roi_heads.training:
                    proposals_with_gt = STrainer.model.roi_heads.label_and_sample_proposals(proposals, targets)
                    #  proposals_with_gt: [Instance,Instance] size = 128
                    #  pls0 = proposals_with_gt[0]              
                    #  pls0.proposal_boxes.tensor.to("cpu").numpy()  for .boxpooler
                    #  pls0.objectness_logits
                    #  pls0.gt_classes
                    #  pls0.gt_boxes
                    #  pls0.mask
#                    detector_losses = STrainer.model.roi_heads._forward_box(features, proposals_with_gt)
                    
                    self = STrainer.model.roi_heads
                    features_roi = [features[f] for f in self.box_in_features]
                    #  froi = features_roi[0].to("cpu").detach().numpy()
                    box_features_p = self.box_pooler(features_roi, [x.proposal_boxes for x in proposals_with_gt])
                    #  bfp = box_features_p.to("cpu").detach().numpy()  [256,256,7,7]
                    box_features_h = self.box_head(box_features_p)
                    #  bfh = box_features_h.to("cpu").detach().numpy()  [256,1024]
                    predictions = self.box_predictor(box_features_h)
                    #  pred = predictions[0].to("cpu").detach().numpy() [256,3] 2 object categories and 1 background class
                    #  pred = predictions[0].to("cpu").detach().numpy() [256,8] bounding box regression deltas for each box. Shape is shape (N,Kx4)
                    if self.training:
                        detector_losses = self.box_predictor.losses(predictions, proposals_with_gt)
                    else:
                        pred_instances, _ = self.box_predictor.inference(predictions, proposals)

                    ### corner_head for train
#                    loss_mask = STrainer.model.roi_heads._forward_mask(features, proposals_with_gt)
                    from detectron2.modeling.roi_heads import select_foreground_proposals
                    if self.training:
                        instances_fg, _ = select_foreground_proposals(proposals_with_gt, self.num_classes)
                    if self.corner_pooler is not None:
                        features = [features[f] for f in self.corner_in_features] 
                        boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances_fg]
                        feats = self.corner_pooler(features, boxes)            # corner_pooler to 28*28
                        #  feats_arr = feats.to("cpu").detach().numpy()                
                    loss_corner = self.corner_head(feats, instances_fg)
                    
                    detector_losses.update(loss_corner)
                else:
                    pred_instances = STrainer.model.roi_heads._forward_box(features, proposals_with_gt)
                    pred_instances = STrainer.model.roi_heads.forward_with_given_boxes(features, pred_instances)

                loss_dict = {}
                loss_dict.update(detector_losses)
                loss_dict.update(proposal_losses)

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
    
############################################################################

    # with open('output/metrics.json','r') as f:
    #     metriclist = []
    #     for line in f.readlines():
    #         metric = json.loads(line)
    #         metriclist.append(metric)
            
    # poly_loss = [i['poly_loss'] for i in metriclist]
    # plt.plot(poly_loss)

    
    










