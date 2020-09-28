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
import test as tester
import sys
import timeit
import torch.autograd
import helper as helper
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import yolo_function as yolo_function


hyperparameters={'lr': 0.0001, 
                 'epochs': 90,
                 'resume_from':0,
                 'coco_version': '2014', #can be either '2014' or '2017'
                 'batch_size': 16,
                 'weight_decay': 0.0005,
                 'momentum': 0.9, 
                 'optimizer': 'sgd', 
                 'alpha': 0.6, 
                 'gamma': 1.3, 
                 'lcoord': 5,
                 'lno_obj': 0.5,
                 'iou_type': (1, 0, 0),
                 'iou_ignore_thresh': 0.213, 
                 'tfidf': True, 
                 'idf_weights': True, 
                 'tfidf_col_names': ['img_freq', 'none', 'none', 'none', 'no_softmax'],
                 'inf_confidence':0.01,
                 'inf_iou_threshold':0.5,
                 'augment': 1, 
                 'workers': 4,
                 'pretrained':False,
                 'path': 'yolo2014', 
                 'reduction': 'sum'}

mode={'bayes_opt':False,
      'debugging':False,
      'show_output':False,
      'multi_gpu':True,
      'show_temp_summary':False,
      'save_summary': True
     }

print(hyperparameters)
if isinstance(hyperparameters['idf_weights'],pd.DataFrame)==False:
    if (hyperparameters['idf_weights']==True):
        hyperparameters['idf_weights']=pd.read_csv('../idf.csv')
    else:
        hyperparameters['idf_weights']=False
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using: ',device)

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

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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
    
    
if(mode['show_temp_summary']==True):
    writer = SummaryWriter('../results/test_vis/')



if hyperparameters['augment']>0:
    train_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([Augment(hyperparameters['augment']),ResizeToTensor(inp_dim)]))
else:
    train_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([ResizeToTensor(inp_dim)]))
    
dataset_len=(len(train_dataset))
batch_size=hyperparameters['batch_size']
mAP_best=0

scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max',patience=5)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                            shuffle=True,collate_fn=helper.collate_fn, num_workers=hyperparameters['workers'])

for i in range(hyperparameters['epochs']):
    outcome=yolo_function.train_one_epoch(model,optimizer,train_dataloader,hyperparameters,mode)
    mAP=test.evaluate(model, device,coco_version,confidence=hyperparameters['inf_confidence'],iou_threshold=hyperparameters['inf_iou_threshold'])
    scheduler.step(mAP)
    if(mode['save_summary']==True):
        writer = SummaryWriter('../results/'+hyperparameters['path'])
        
        writer.add_scalar('Loss/train', outcome['avg_loss'], hyperparameters['resume_from'])
        writer.add_scalar('AIoU/train', outcome['avg_iou'], hyperparameters['resume_from'])
        writer.add_scalar('PConf/train', outcome['avg_conf'], hyperparameters['resume_from'])
        writer.add_scalar('NConf/train', outcome['avg_no_conf'], hyperparameters['resume_from'])
        writer.add_scalar('PClass/train', outcome['avg_pos'], hyperparameters['resume_from'])
        writer.add_scalar('NClass/train', outcome['avg_neg'], hyperparameters['resume_from'])
        
        writer.add_scalar('mAP/valid', mAP, hyperparameters['resume_from'])

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['total_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'epoch':hyperparameters['resume_from']+1,
            'mAP': mAP
            }, PATH+hyperparameters['path']+'.tar')

        
    if mAP>mAP_best:
        torch.save(checkpoint, PATH+hyperparameters['path']+'_best.tar')
        mAP_best=mAP
        
    