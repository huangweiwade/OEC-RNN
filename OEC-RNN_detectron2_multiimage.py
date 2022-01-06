# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 18:58:26 2021

@author: Huangwei
"""
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
import numpy as np
import os, json, cv2, random, torch
from glob import glob
from tqdm import tqdm
import time
import math
from scipy import ndimage as ndi

from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.engine import DefaultPredictor


import matplotlib;matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Rectangle,Circle
from matplotlib.collections import PatchCollection

import Utils.utils as utils
import skimage.transform as transform
from skimage import io
from skimage import measure

import sys
sys.path.append('/home/super/桌面/polyrnn-pp-pytorch/code')
from Poly import polyrnnpp
from Evaluation import metrics
from sklearn.metrics import f1_score
import torch.nn.functional as F
import re

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generateColor(Num):
    print('generateColor:',Num)
    ec = []; fc = []
    for i in range(Num):
        rgb = np.random.random((3,))*0.6+0.4
        c = (rgb).tolist()
        ec.append(c) 
        ca = np.append(rgb,0.5).tolist()
        fc.append(ca)
    return ec,fc
# ec = np.random.random((3,))*0.6+0.4;fc = np.append(ec,0.5)
def austin_building(main_dir):
    json_filename = glob(os.path.join(main_dir,'annotation.json'))[0]
    with open(json_filename,'r') as f:
        dataset_dicts = json.load(f)
    for d in dataset_dicts:
        for ann in d['annotations']: # ann=d['annotations'][0]["bbox_mode"]
            ann["bbox_mode"] = BoxMode.XYWH_ABS
    return dataset_dicts

def extract_crop_by_boxes(img,boxes):  # img=utils.rgb_img_read(file_path)
    context_expansion = random.uniform(0.3,0.3)
    crops = []
    for i,bbox in enumerate(boxes):
        x0, y0, x1, y1 = list(bbox); w=x1-x0;h=y1-y0  
        '''
        bb=[Rectangle((x0,y0),w,h)]
        plt.imshow(img)
        ax = plt.gca()
        b = PatchCollection(bb,facecolor='None',edgecolors='r',linewidths=3 ,linestyle="dashed")#边
        ax.add_collection(b)
        '''
        l=min(h,w)
        x_min = int(torch.floor(x0 - l*( context_expansion)/2.));x_max = int(torch.ceil(x1 + l*( context_expansion)/2.))
        x_min = max(0, x_min);x_max = min(img.shape[1] - 1, x_max)
        y_min = int(torch.floor(y0 - l*(context_expansion)/2.));y_max = int(torch.ceil(y1 + l*( context_expansion)/2.))

        y_min = max(0, y_min);y_max = min(img.shape[0] - 1, y_max)
        w0=x_max-x_min; h0=y_max-y_min
        '''
        bb=[Rectangle((x_min,y_min),w0,h0)]
        plt.imshow(img)
        ax = plt.gca()
        b = PatchCollection(bb,facecolor='None',edgecolors='r',linewidths=3 ,linestyle="dashed")#边
        ax.add_collection(b)
        '''
        patch_img = img[y_min:y_max, x_min:x_max, :]
        try:
            feature = transform.resize(patch_img,(224,224))
        except:
            continue
        starting_point = [x_min, y_min]
        feature = np.expand_dims(feature.transpose(2,0,1),axis=0)
        outdict = {
                    'feat': feature,
                    'starting_point': starting_point,
                    'h': int(h0),
                    'w': int(w0),
                    }
        crops.append(outdict)
    return crops

def process_outputs(datasets,outputs,shape):  
    fs=[];polys=[];v_atts=[];e_atts=[];ori_polys=[]
    mask=np.zeros(shape[:2], dtype=np.uint8)
    vertex_attention = np.zeros(shape[:2], dtype=np.float32)
    edge_attention = np.zeros(shape[:2], dtype=np.float32)
    for data,p in zip(datasets,outputs):
        f = data['feat'][0].transpose(1,2,0)
        f = transform.resize(f,(data['h'],data['w']))
        fs.append(f)
        poly = p['pred_polys']
        if np.max(poly) == 784:
            poly = poly[0,:np.argmax(poly)]
        else:
            poly = poly[0,:]
        poly = utils.class_to_xy(poly, 28)
        poly = utils.poly0g_to_poly01(poly, 28)
        poly[:,0] *= data['w']
        poly[:,1] *= data['h']
        ori_polys.append((poly+0.5).astype(np.int))
        poly[:,0] += data['starting_point'][0]
        poly[:,1] += data['starting_point'][1]
        poly = (poly+0.5).astype(np.int)

        v_logprobs = F.log_softmax(torch.tensor(p['vertex_logits']), -1).cpu().numpy().reshape(28,28)
        vertex_att = transform.resize(v_logprobs,(data['h'],data['w']))
        # vertex_attention[data['starting_point'][1]:data['starting_point'][1]+data['h'],
        #                data['starting_point'][0]:data['starting_point'][0]+data['w']] = vertex_att
        v_atts.append(vertex_att)

        e_logprobs = F.log_softmax(torch.tensor(p['edge_logits']), -1).cpu().numpy().reshape(28,28)
        edge_att = transform.resize(e_logprobs,(data['h'],data['w']))
        # edge_attention[data['starting_point'][1]:data['starting_point'][1]+data['h'],
        #                data['starting_point'][0]:data['starting_point'][0]+data['w']] = edge_att
        e_atts.append(edge_att)
        
        cv2.fillPoly(mask, [poly], 255)
        polys.append(poly)
    return fs,ori_polys,polys,mask,v_atts,e_atts,vertex_attention,edge_attention
            
def Cropimage(image_paths,image_shape):
    out_dir = image_paths[0].split('.')[0]
    outimg_dir = os.path.join(out_dir,'Crop_images_{}'.format(image_shape[0]))
    if not os.path.exists(outimg_dir):
        os.makedirs(outimg_dir)
    ## *** Clip the img to x00*x00 ***  
    for image_file in image_paths:
        img = cv2.imread(image_file)
        img = cv2.resize(img, (int(img.shape[1]*1), int(img.shape[0]*1)))
        pad_h = (math.ceil(img.shape[0]/image_shape[0]))*image_shape[0]
        pad_w = (math.ceil(img.shape[1]/image_shape[1]))*image_shape[1]
        pad_img = np.zeros((pad_h,pad_w,3),dtype='uint8')
        pad_img[:img.shape[0],:img.shape[1],:]=img #填充影像，使得可滑窗
        image = pad_img
        
        row = image.shape[0]
        col = image.shape[1]
        for i in range(int(row/image_shape[0])):
            for j in range(int(col/image_shape[1])):
                subimage = image[i*image_shape[0]:(i+1)*image_shape[0],j*image_shape[1]:(j+1)*image_shape[1],:]
                cv2.imwrite(outimg_dir+'/'+os.path.basename(image_file).split('.')[0]+'_'+'%02d'%(i+1)+'_'+'%02d'%(j+1)+'.tif',subimage)      
    print('crop is finished')
    return outimg_dir
    
def Mosaicimage(output_dir,crop_shape,shape):
    image_paths = glob(output_dir+'/*.*')
    ## *** Mosaic the img  ***  
    maxrow = 0; maxcol = 0
    for image_file in tqdm(image_paths):
        img=cv2.imread(image_file)
        query=re.compile("_\d\d_\d\d").findall(image_file)[0]
        
        row=int(query[1:3]);    col=int(query[4:6])
        if row > maxrow :maxrow = row
        if col > maxcol :maxcol = col
        
    out_img = np.zeros([crop_shape[0]*maxrow,crop_shape[1]*maxcol,3])
    for image_file in image_paths:
        img=cv2.imread(image_file)
        query=re.compile("_\d\d_\d\d").findall(image_file)[0]
        row=int(query[1:3]);    col=int(query[4:6])
        row_start=800*(row-1);  col_start=800*(col-1)
        out_img[row_start:row_start+800,col_start:col_start+800,:]+=img
    out_image = out_img[:shape[0],:shape[1],0]
    cv2.imwrite(output_dir+"_mosaic.png",out_image)
    print('mosaic is finished')
    return out_image

           
if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("detectron2/configs/COCO-InstanceSegmentation/corner_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.MASK_ON = True
    cfg.DATASETS.TEST = ("austin_ValForDetectron2", )
    cfg.MODEL.WEIGHTS = '/home/super/桌面/OEC-RNN/model/model_0019999.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.1
    
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.RPN.NMS_THRESH = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 
    cfg.exp = '/home/super/桌面/OEC-RNN/Experiments/mle.json'
    cfg.reload = '/home/super/桌面/OEC-RNN/model/epoch16_step47000.pth'
    D = DefaultPredictor(cfg)
    D.model.eval()
    opts = json.load(open(cfg.exp, 'r'));opts['mode'] = 'test'
    D.corner_head = polyrnnpp.OECRNN(opts).to(device)
    D.corner_head.reload(cfg.reload, strict=False)
    
#############################################################################

    file_path = ['/home/super/桌面/OEC-RNN/test_image/wurenjicorrect_clip_image.tif']  ## 测试文件
    
    img = cv2.imread(file_path[0])
    shape = img.shape[:2]
    file_name = os.path.basename(file_path[0]).split('.')[0]
    output_dir = 'Output/{}_cropmask_pred_{:.3f}'.format(file_name,cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    crop_shape = [800,800]
    cropdir = Cropimage(file_path,crop_shape)
    filelist = glob(cropdir+'/*.*')
    for file_path in tqdm(filelist):
#        file_path = '/home/super/桌面/detectron2/Yushu/WuRenJi_crop.tif'
        f_name = os.path.basename(file_path).split('.')[0]
        im = cv2.imread(file_path)#[2400:3200,4000:4800] #[4900:5700,5600:6400] 
        with torch.no_grad(): 
            i = torch.as_tensor(D.aug.get_transform(im).apply_image(im).astype("float32").transpose(2, 0, 1))
            images = D.model.preprocess_image([{"image": i, "height": i.shape[0], "width": i.shape[1]}])
            features = D.model.backbone(images.tensor)
            proposals, _= D.model.proposal_generator(images, features, None)
            
            boxes = [x.pred_boxes for x in D.model.roi_heads._forward_box(features, proposals)][0]
            img_crop = extract_crop_by_boxes(utils.rgb_img_read(file_path),boxes)
            outputs=[]
            for f in img_crop:
                output = D.corner_head(torch.tensor(f['feat'],dtype=torch.float32).to(device),
                                       fp_beam_size=1, 
                                       lstm_beam_size=1)
                for k in output.keys(): output[k] = output[k].cpu().numpy()
                outputs.append(output)
        _,_,_,mask,_,_,_,_ = process_outputs(img_crop,outputs,im.shape)
        io.imsave(output_dir+'/{}.png'.format(f_name),mask)
    mask_result = Mosaicimage(output_dir,crop_shape,shape) #拼接合成
  
