from dataset import *
import timeit 
import cv2
import numpy as np
from zipfile import ZipFile
import pandas as pd
import torch
import time
from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import test as tester
import sys
import timeit
import torch.autograd
import helper as helper
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


def train_yolo(weight_decay,momentum,gamma,alpha,lcoord,lno_obj,iou_ignore_thresh,iou_type,tf_idf):
    
    iou_type=int(iou_type)
    if(iou_type)==0:
        iou_type=(0,0,0)
    elif(iou_type==1):
        iou_type=(1,0,0)
    elif(iou_type==2):
        iou_type=(0,1,0)
    else:
        iou_type=(0,0,1)
    
    tf_idf=int(round(tf_idf))
    if tf_idf==0:
        tf_idf=False
    else:
        tf_idf=True
    
    hyperparameters={'lr':0.0001,
                 'epochs':10,
                 'coco_version':'2014',
                 'batch_size':16,
                 'weight_decay':weight_decay,
                 'momentum':0.9,
                 'optimizer':'sgd',
                 'alpha':alpha,
                 'gamma':gamma,
                 'lcoord':lcoord,
                 'lno_obj':lno_obj,
                 'iou_type':iou_type,#(GIoU,DIoU,CIoU) default is 0,0,0 for iou
                 'iou_ignore_thresh':iou_ignore_thresh,
                 'tfidf':tf_idf,
                 'idf_weights':True,
                 'tfidf_col_names':['obj_freq','area','xc','yc','no_softmax'], #default is ['obj_freq/img_freq','area','xc','yc','softmax']-->[class_weights,scale_weights,xweights,yweights,softmax/no_softmax]
                 'augment':0,
                 'workers':4,
                 'path':'bayesian_opt',
                 'reduction':'sum'}
    coco_version=hyperparameters['coco_version']
    net = Darknet("../cfg/yolov3.cfg")
    inp_dim=net.inp_dim
    pw_ph=net.pw_ph.to(device='cuda')
    cx_cy=net.cx_cy.to(device='cuda')
    stride=net.stride.to(device='cuda')

    print(hyperparameters)

    if (hyperparameters['idf_weights']==True):
        hyperparameters['idf_weights']=pd.read_csv('../idf.csv')
    else:
        hyperparameters['idf_weights']=False

    '''
    when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
    if you start fresh then first model.load_weights and then make it parallel
    '''
    try:
        PATH = '../pth/'+hyperparameters['path']+'/'
        weights = torch.load(PATH+hyperparameters['path']+'_best.pth')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we https://pytorch.org/docs/stable/data.html#torch.utils.data.Datasetare on a CUDA machine, this should print a CUDA device:
        net.to(device)

        if (torch.cuda.device_count() > 1):
            model = nn.DataParallel(net)
            model.to(device)
            model.load_state_dict(weights)
        else:
            model=net
            model.to(device)
            model.load_state_dict(weights)

    except FileNotFoundError:
        
        net.load_weights("../yolov3.weights")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Assuming that we are on a CUDA machine, this should print a CUDA device:

        net.to(device)
        if (torch.cuda.device_count() > 1):
            model = nn.DataParallel(net)
            model.to(device)
        else:
            model=net

    if hyperparameters['augment']>0:
        transformed_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([Augment(hyperparameters['augment']),ResizeToTensor(inp_dim)]))
    else:
        transformed_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([ResizeToTensor(inp_dim)]))

    dataset_len=(len(transformed_dataset))
    batch_size=hyperparameters['batch_size']
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=False,collate_fn=helper.my_collate, num_workers=hyperparameters['workers'])


    if hyperparameters['optimizer']=='sgd':
        optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
    elif hyperparameters['optimizer']=='adam':
        optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

    lambda1 = lambda epoch: 0.95**epoch
    scheduler=optim.lr_scheduler.LambdaLR(optimizer, lambda1, last_epoch=-1)
    mAP_max=0
    epochs=hyperparameters['epochs']
    for e in range(epochs):
        total_loss=0
        write=0
        misses=0
        avg_iou=0
        prg_counter=0
        train_counter=1
        avg_conf=0
        avg_no_conf=0
        avg_pos=0
        avg_neg=0
        for images,targets,img_names in dataloader:
            optimizer.zero_grad()
        #         targets,anchors,offset,strd,mask=helper.collapse_boxes(targets,pw_ph,cx_cy,stride)
            images=images.cuda()
            raw_pred = model(images, torch.cuda.is_available())
        #         raw_pred=helper.expand_predictions(raw_pred,mask)
            true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)

            targets,anchors,offset,strd,mask=helper.collapse_boxes(targets,pw_ph,cx_cy,stride)
            targets=targets.cuda()
            fall_into_mask=util.get_fall_into_mask(targets,offset,strd,inp_dim)
            resp_raw_pred,resp_true_pred,resp_anchors,resp_offset,resp_strd=util.build_tensors(raw_pred,true_pred,anchors,offset,strd,fall_into_mask,mask)

            iou,iou_mask=util.get_iou_mask(targets,resp_true_pred,inp_dim,hyperparameters)
            no_obj=util.get_noobj(true_pred,targets,fall_into_mask,mask,hyperparameters,inp_dim)
            resp_raw_pred=resp_raw_pred[iou_mask]
            resp_anchors=resp_anchors[iou_mask]
            resp_offset=resp_offset[iou_mask]
            resp_strd=resp_strd[iou_mask]
            
            try:
                loss=util.yolo_loss(resp_raw_pred,targets,no_obj,mask,resp_anchors,resp_offset,resp_strd,inp_dim,hyperparameters)
            except RuntimeError:
                break
            
            try:
                loss.backward()
                optimizer.step()
            except RuntimeError:
                print('\nthe sum is:',sum(mask))

        mAP=tester.get_map(model,confidence=0.01,iou_threshold=0.5)
        if mAP>mAP_max:
            mAP_max=mAP
        else:
            break
    print(mAP_max)
    return mAP_max