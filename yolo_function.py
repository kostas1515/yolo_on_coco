from dataset import *
from darknet import *
import numpy as np
import pandas as pd
import torch
import util as util
import torch.optim as optim
import sys
import torch.autograd
import helper as helper
from torch import autograd
from torch.utils.tensorboard import SummaryWriter
import test as tester


def train_one_epoch(model,optimizer,dataloader,hyperparameters,mode):
    
#     iou_type=int(iou_type)
#     if(iou_type)==0:
#         iou_type=(0,0,0)
#     elif(iou_type==1):
#         iou_type=(1,0,0)
#     elif(iou_type==2):
#         iou_type=(0,1,0)
#     else:
#         iou_type=(0,0,1)        
    model.train()
    
    if(mode['show_temp_summary']==True):
        writer = SummaryWriter('../results/test_vis/')
    epoch=hyperparameters['resume_from']
    
    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
        pw_ph=model.module.pw_ph
        cx_cy=model.module.cx_cy
        stride=model.module.stride
    else:
        inp_dim=model.inp_dim
        pw_ph=model.pw_ph
        cx_cy=model.cx_cy
        stride=model.stride
    
    coco_version=hyperparameters['coco_version']
    
    inf_confidence=hyperparameters['inf_confidence']
    inf_iou_threshold=hyperparameters['inf_iou_threshold']
    
    
    pw_ph=pw_ph.cuda()
    cx_cy=cx_cy.cuda()
    stride=stride.cuda()



    
        
    dataset_len=len(dataloader.dataset)
    batch_size=dataloader.batch_size
    total_loss=0
    avg_iou=0
    prg_counter=0
    train_counter=0
    avg_conf=0
    avg_no_conf=0
    avg_pos=0
    avg_neg=0
    for images,targets,img_names in dataloader:
        train_counter=train_counter+1
        optimizer.zero_grad()
        images=images.cuda()
        
        if mode['debugging']==True:
            with autograd.detect_anomaly():
                raw_pred = model(images, torch.cuda.is_available())
        else:
            raw_pred = model(images, torch.cuda.is_available())
            if (torch.isinf(raw_pred).sum()>0):
                return 0
                break
            

        true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)

        targets,anchors,offset,strd,mask=helper.collapse_boxes(targets,pw_ph,cx_cy,stride)
        targets=targets.cuda()
        fall_into_mask=util.get_fall_into_mask(targets,offset,strd,inp_dim)
        resp_raw_pred,resp_true_pred,resp_anchors,resp_offset,resp_strd=util.build_tensors(raw_pred,true_pred,anchors,offset,strd,fall_into_mask,mask)

        iou,iou_mask=util.get_iou_mask(targets,resp_true_pred,inp_dim,hyperparameters)
        iou=iou.T.max(dim=1)[0].mean().item()
        no_obj_mask=util.get_noobj(true_pred,targets,fall_into_mask,mask,hyperparameters,inp_dim)
        k=0
        no_obj=[]
        for f,i in no_obj_mask:
            no_obj.append(raw_pred[k,f][i][:,4])
            k=k+1
        no_obj=torch.cat(no_obj)
        no_obj_conf=torch.sigmoid(no_obj.clone().detach()).mean().item()
        
        
        resp_raw_pred=resp_raw_pred[iou_mask]
        resp_anchors=resp_anchors[iou_mask]
        resp_offset=resp_offset[iou_mask]
        resp_strd=resp_strd[iou_mask]
        conf=resp_true_pred[iou_mask][:,4].mean().item()
        class_mask=targets[:,5:].type(torch.BoolTensor).squeeze(0)
        
        if(iou_mask.sum()==class_mask.shape[0]):
            pos_class=resp_true_pred[iou_mask][:,5:][class_mask].mean().item()
            neg_class=resp_true_pred[iou_mask][:,5:][~class_mask].mean().item()
        else:
            pos_class=0
            neg_class=0
        if mode['debugging']==True:
            with autograd.detect_anomaly():
                loss=util.yolo_loss(resp_raw_pred,targets,no_obj,mask,resp_anchors,resp_offset,resp_strd,inp_dim,hyperparameters)
        elif mode['bayes_opt']==True:
            try:
                loss=util.yolo_loss(resp_raw_pred,targets,no_obj,mask,resp_anchors,resp_offset,resp_strd,inp_dim,hyperparameters)
            except RuntimeError:
#                 print('bayes opt failed')
                return 0
                
        else:
            loss=util.yolo_loss(resp_raw_pred,targets,no_obj,mask,resp_anchors,resp_offset,resp_strd,inp_dim,hyperparameters)
        loss.backward()
        optimizer.step()
        
        if mode['show_output']==True:
            sys.stdout.write('\rPgr:'+str(prg_counter/dataset_len*100*batch_size)+'%' ' L:'+ str(loss.item()))
            sys.stdout.write(' IoU:' +str(iou)+' pob:'+str(conf)+ ' nob:'+str(no_obj_conf))
            sys.stdout.write(' PCls:' +str(pos_class)+' ncls:'+str(neg_class))
            sys.stdout.flush()
        
        avg_conf=avg_conf+conf
        avg_no_conf=avg_no_conf+no_obj_conf
        avg_pos=avg_pos+pos_class
        avg_neg=avg_neg+neg_class
        total_loss=total_loss+loss.item()
        avg_iou=avg_iou+iou
        prg_counter=prg_counter+1
        
        
            
        
        if(mode['show_temp_summary']==True):
            writer.add_scalar('AvLoss/train', total_loss/train_counter, train_counter)
            writer.add_scalar('AvIoU/train', avg_iou/train_counter, train_counter)
            writer.add_scalar('AvPConf/train', avg_conf/train_counter, train_counter)
            writer.add_scalar('AvNConf/train', avg_no_conf/train_counter, train_counter)
            writer.add_scalar('AvClass/train', avg_pos/train_counter, train_counter)
            writer.add_scalar('AvNClass/train', avg_neg/train_counter, train_counter)
            
            
    
    total_loss = total_loss/train_counter
    avg_iou = avg_iou/train_counter
    avg_pos = avg_pos/train_counter
    avg_neg = avg_neg/train_counter
    avg_conf = avg_conf/train_counter
    avg_no_conf = avg_no_conf/train_counter
    
    
    
    
    if mode['bayes_opt']==False:
        mAP=tester.get_map(model,confidence=0.01,iou_threshold=0.5,coco_version=coco_version,subset=1)
        
        torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_loss': total_loss,
        'avg_iou': avg_iou,
        'avg_pos': avg_pos,
        'avg_neg':avg_neg,
        'avg_conf': avg_conf,
        'avg_no_conf': avg_no_conf,
        'epoch':epoch+1,
        'mAP':mAP
        }, '../pth/'+hyperparameters['path']+'/'+hyperparameters['path']+'.tar')
        
        return mAP
    
    else:
        mAP=tester.get_map(model,confidence=inf_confidence,iou_threshold=inf_iou_threshold,coco_version=coco_version,subset=0.18)
        
        return mAP
    
    
    
    

    