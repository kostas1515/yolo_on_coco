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
import test


def train_one_epoch(model,optimizer,dataloader,hyperparameters,mode):
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
    

    
    pw_ph=pw_ph.cuda()
    cx_cy=cx_cy.cuda()
    stride=stride.cuda()



    
    break_flag=0
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
    for images,targets in dataloader:
        train_counter=train_counter+1
        prg_counter=prg_counter+1
        optimizer.zero_grad()
        images=images.cuda()
        
        if mode['debugging']==True:
            with autograd.detect_anomaly():
                raw_pred = model(images, torch.cuda.is_available())
        else:
            raw_pred = model(images, torch.cuda.is_available())
            if (torch.isinf(raw_pred).sum()>0):
                break_flag=1
                break
            

        true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)
        iou_list=util.get_iou_list(true_pred, targets,hyperparameters,inp_dim)

        resp_raw_pred,resp_cx_cy,resp_pw_ph,resp_stride,no_obj=util.build_tensors(raw_pred,iou_list,pw_ph,cx_cy,stride,hyperparameters)
        
        
        stats=helper.get_progress_stats(true_pred,no_obj,iou_list,targets)

        if hyperparameters['wasserstein']==True:
            no_obj=util.get_wasserstein_matrices(raw_pred,iou_list,inp_dim)
        
        if mode['debugging']==True:
            with autograd.detect_anomaly():
                loss=util.yolo_loss(resp_raw_pred,targets,no_obj,resp_pw_ph,resp_cx_cy,resp_stride,inp_dim,hyperparameters)
        elif mode['bayes_opt']==True:
            try:
                loss=util.yolo_loss(resp_raw_pred,targets,no_obj,resp_pw_ph,resp_cx_cy,resp_stride,inp_dim,hyperparameters)
            except RuntimeError:
#                 print('bayes opt failed')
                break_flag=1
                break     
        else:
            loss=util.yolo_loss(resp_raw_pred,targets,no_obj,resp_pw_ph,resp_cx_cy,resp_stride,inp_dim,hyperparameters)
        loss.backward()
        optimizer.step()
        
        
        avg_conf=avg_conf+stats['pos_conf']
        avg_no_conf=avg_no_conf+stats['neg_conf']
        avg_pos=avg_pos+stats['pos_class']
        avg_neg=avg_neg+stats['neg_class']
        total_loss=total_loss+loss.item()
        avg_iou=avg_iou+stats['iou']
        
        
        if mode['show_output']==True:
            sys.stdout.write('\rPgr:'+str(prg_counter/dataset_len*100*batch_size)+'%' ' L:'+ str(loss.item()))
            sys.stdout.write(' IoU:' +str(stats['iou'])+' pob:'+str(stats['pos_conf'])+ ' nob:'+str(stats['neg_conf']))
            sys.stdout.write(' PCls:' +str(stats['pos_class'])+' ncls:'+str(stats['neg_class']))
            sys.stdout.flush()
              
        
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
    
    
    
    

    outcome = {
    'avg_loss': total_loss,
    'avg_iou': avg_iou,
    'avg_pos': avg_pos,
    'avg_neg':avg_neg,
    'avg_conf': avg_conf,
    'avg_no_conf': avg_no_conf,
    'broken': break_flag
    }
    

        
    return outcome
    
    
    
    
    

    