from dataset import *
import timeit 
import cv2
import numpy as np
from zipfile import ZipFile
import pandas as pd
import torch
import time
from darknet import *
import util as util
import torch.optim as optim
import test 
import sys
import timeit
import torch.autograd
import helper as helper
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import test
import yolo_function as yolo_function


def bayesian_opt(lr,w,m,g,a,lcoor,lno,iou_thresh,iou_type,inf_c,inf_t,bayes_opt=True):
    
    iou_type=int(round(iou_type))
    if(iou_type)==0:
        iou_type=(0,0,0)
    elif(iou_type==1):
        iou_type=(1,0,0)
    elif(iou_type==2):
        iou_type=(0,1,0)
    else:
        iou_type=(0,0,1) 
    
    hyperparameters={'lr': lr, 
                     'epochs': 1,
                     'resume_from':0,
                     'coco_version': '2014', #can be either '2014' or '2017'
                     'batch_size': 16,
                     'weight_decay': w,
                     'momentum': m, 
                     'optimizer': 'sgd', 
                     'alpha': a, 
                     'gamma':g, 
                     'lcoord': lcoor,
                     'lno_obj': lno,
                     'iou_type': iou_type,
                     'iou_ignore_thresh': iou_thresh,
                     'inf_confidence':inf_c,
                     'inf_iou_threshold':inf_t,
                     'wasserstein':False,
                     'tfidf': True, 
                     'idf_weights': True, 
                     'tfidf_col_names': ['img_freq', 'none', 'none', 'none', 'no_softmax'],
                     'augment': 1, 
                     'workers': 4,
                     'pretrained':True,
                     'path': 'yolo2014', 
                     'reduction': 'sum'}

    mode={'bayes_opt':bayes_opt,
          'debugging':False,
          'show_output':False,
          'multi_gpu':False,
          'show_temp_summary':False,
          'save_summary': bayes_opt==False
         }

#     print(hyperparameters)
    if isinstance(hyperparameters['idf_weights'],pd.DataFrame)==False:
        if (hyperparameters['idf_weights']==True):
            hyperparameters['idf_weights']=pd.read_csv('../idf.csv')
        else:
            hyperparameters['idf_weights']=False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print('Using: ',device)

    net = Darknet("../cfg/yolov3.cfg")
    coco_version=hyperparameters['coco_version']
    inp_dim=net.inp_dim

    '''
    when loading weights from dataparallel model then, you first need to instatiate the dataparallel model 
    if you start fresh then first model.load_weights and then make it parallel
    '''
    try:
        PATH = '../pth/'+hyperparameters['path']+'/'
        checkpoint = torch.load(PATH+hyperparameters['path']+'.tar')
                    # Assuming that we https://pytorch.org/docs/stable/data.html#torch.utils.data.Datasetare on a CUDA machine, this should print a CUDA device:
        net.to(device)

        if (torch.cuda.device_count() > 1)&(mode['multi_gpu']==True):
            model = nn.DataParallel(net)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
        else:
            model=net
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)

        if hyperparameters['optimizer']=='sgd':
            optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
        elif hyperparameters['optimizer']=='adam':
            optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])

#         optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        hyperparameters['resume_from']=checkpoint['epoch']

    except FileNotFoundError:
        if (hyperparameters['pretrained']==True):
            print("WARNING FILE NOT FOUND INSTEAD USING OFFICIAL PRETRAINED")
            net.load_weights("../yolov3.weights")

            net.to(device)
            if (torch.cuda.device_count() > 1)&(mode['multi_gpu']==True):
                model = nn.DataParallel(net)
                model.to(device)
            else:
                model=net

            if hyperparameters['optimizer']=='sgd':
                optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
            elif hyperparameters['optimizer']=='adam':
                optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
            hyperparameters['resume_from']=0
        else:
            try:
                PATH = '../pth/'+hyperparameters['path']+'/'
                os.mkdir(PATH)
            except FileExistsError:
                pass
#                     print('path already exist')

            if (torch.cuda.device_count() > 1)&(mode['multi_gpu']==True):
                model = nn.DataParallel(net)
                model.to(device)
            else:
                model=net
                model.to(device)

            if hyperparameters['optimizer']=='sgd':
                optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
            elif hyperparameters['optimizer']=='adam':
                optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'])
            hyperparameters['resume_from']=0
                
            
            

    if bayes_opt==True:
        tr_subset=0.01
        ts_subset=0.1
    else:
        tr_subset=1
        ts_subset=1
    if(mode['save_summary']==True):
        writer = SummaryWriter('../results/'+hyperparameters['path'])
   

    
    if hyperparameters['augment']>0:
        train_dataset=Coco(partition='train',coco_version=coco_version,subset=tr_subset,
                           transform=transforms.Compose([Augment(hyperparameters['augment']),ResizeToTensor(inp_dim)]))
    else:
        train_dataset=Coco(partition='train',coco_version=coco_version,subset=subset,transform=transforms.Compose([ResizeToTensor(inp_dim)]))

    dataset_len=(len(train_dataset))
    batch_size=hyperparameters['batch_size']

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True,collate_fn=helper.collate_fn, num_workers=hyperparameters['workers'])

    for i in range(hyperparameters['epochs']):
        outcome=yolo_function.train_one_epoch(model,optimizer,train_dataloader,hyperparameters,mode)
        
        if outcome['broken']==1:
            return 0
        else:
            mAP=test.evaluate(model, device,coco_version,confidence=hyperparameters['inf_confidence'],iou_threshold=hyperparameters['inf_iou_threshold'],subset=ts_subset)
            if(len(mAP)==0):
                mAP=0
            else:
                mAP=mAP[0]
        if(mode['save_summary']==True):
            
            writer.add_scalar('Loss/train', outcome['avg_loss'], hyperparameters['resume_from'])
            writer.add_scalar('AIoU/train', outcome['avg_iou'], hyperparameters['resume_from'])
            writer.add_scalar('PConf/train', outcome['avg_conf'], hyperparameters['resume_from'])
            writer.add_scalar('NConf/train', outcome['avg_no_conf'], hyperparameters['resume_from'])
            writer.add_scalar('PClass/train', outcome['avg_pos'], hyperparameters['resume_from'])
            writer.add_scalar('NClass/train', outcome['avg_neg'], hyperparameters['resume_from'])
            writer.add_scalar('mAP/valid', mAP, hyperparameters['resume_from'])
            
        if(mode['bayes_opt']==False):
            
            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['avg_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'epoch':hyperparameters['resume_from']+1
            }, PATH+hyperparameters['path']+'.tar')

#             hyperparameters['resume_from']=checkpoint['epoch']+1
                
    return mAP