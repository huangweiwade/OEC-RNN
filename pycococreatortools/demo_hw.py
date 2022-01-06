import numpy as np
import cv2 as cv
from PIL import Image
#from matplotlib import pyplot
import matplotlib.pyplot as plt
import random
#from google.colab.patches import cv2_imshow

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#下载图片
im = cv.imread("/home/super/桌面/detectron2/000000439715.jpg") # plt.imshow(im)

cfg = get_cfg()
cfg.merge_from_file("/home/super/桌面/detectron2/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  #模型阈值
cfg.MODEL.WEIGHTS = "/home/super/桌面/detectron2/model_final_f10217.pkl"
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

pred_classes = outputs["instances"].pred_classes
print(pred_classes)
pred_boxes = outputs["instances"].pred_boxes
print(pred_boxes)
pred_masks = np.array(outputs["instances"].pred_masks.to("cpu"))
scores = np.array(outputs["instances"].scores.to("cpu"))

img = im[:, :, ::-1] # BGR 转 RGB # plt.imshow(img)

 #在原图上画出检测结果
plt.figure()
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
plt.imshow(v.get_image())
#plt.show()
#plt.savefig('result.png')
#plt.close()
