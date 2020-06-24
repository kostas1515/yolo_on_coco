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



net = Darknet("../cfg/yolov3.cfg")
inp_dim=net.inp_dim
pw_ph=net.pw_ph.to(device='cuda')
cx_cy=net.cx_cy.to(device='cuda')
stride=net.stride.to(device='cuda')


hyperparameters={'lr':0.0001,
                 'epochs':30,
                 'batch_size':16,
                 'weight_decay':0.001,
                 'momentum':0.9,
                 'optimizer':'sgd',
                 'alpha':0.5,
                 'gamma':0,
                 'lcoord':5,
                 'lno_obj':1,
                 'iou_type':(0,0,0),#(GIoU,DIoU,CIoU) default is 0,0,0 for iou
                 'iou_ignore_thresh':0.5,
                 'tfidf':True,
                 'idf_weights':True,
                 'tfidf_col_names':['obj_freq','area','xc','yc','softmax'], #default is ['obj_freq/img_freq','area','xc','yc','softmax']-->[class_weights,scale_weights,xweights,yweights,softmax/no_softmax]
                 'augment':0,
                 'workers':4,
                 'path':'tfidf_soft_b16_v2',
                 'reduction':'sum'}

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
    PATH = '../'+hyperparameters['path']
    weights = torch.load(PATH)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we https://pytorch.org/docs/stable/data.html#torch.utils.data.Datasetare on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    if (torch.cuda.device_count() > 1):
        print("Using ", torch.cuda.device_count(), "GPUs!")
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

    print(device)
    net.to(device)
    if (torch.cuda.device_count() > 1):
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(net)
        model.to(device)
    else:
        model=net
        
if hyperparameters['augment']>0:
    transformed_dataset=Coco(partition='train',transform=transforms.Compose([Augment(hyperparameters['augment']),ResizeToTensor(inp_dim)]))
else:
    transformed_dataset=Coco(partition='train',transform=transforms.Compose([ResizeToTensor(inp_dim)]))
    


writer = SummaryWriter('../results/'+hyperparameters['path'])
dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=hyperparameters['batch_size']
dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=False,collate_fn=helper.my_collate, num_workers=hyperparameters['workers'])


if hyperparameters['optimizer']=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=hyperparameters['lr'], weight_decay=hyperparameters['weight_decay'], momentum=hyperparameters['momentum'])
elif hyperparameters['otimizer']=='adam':
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
    train_counter=0
    avg_conf=0
    avg_no_conf=0
    avg_pos=0
    avg_neg=0
    for images,targets,img_names in dataloader:
        train_counter=train_counter+1
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
        iou=iou.T.max(dim=1)[0].mean().item()
        no_obj=util.get_noobj(true_pred,targets,fall_into_mask,mask,hyperparameters)
        no_obj_conf=no_obj.mean().item()

        resp_raw_pred=resp_raw_pred[iou_mask]
        resp_anchors=resp_anchors[iou_mask]
        resp_offset=resp_offset[iou_mask]
        resp_strd=resp_strd[iou_mask]
        conf=resp_raw_pred[:,4].mean().item()
        class_mask=targets[:,5:].type(torch.BoolTensor).squeeze(0)

        pos_class=resp_raw_pred[:,5:][class_mask].mean().item()
        neg_class=resp_raw_pred[:,5:][~class_mask].mean().item()
        loss=util.yolo_loss(resp_raw_pred,targets,no_obj,mask,resp_anchors,resp_offset,resp_strd,inp_dim,hyperparameters)
        
        loss.backward()
        optimizer.step()
        
        avg_conf=avg_conf+conf
        avg_no_conf=avg_no_conf+no_obj_conf
        avg_pos=avg_pos+pos_class
        avg_neg=avg_neg+neg_class
        total_loss=total_loss+loss.item()
        avg_iou=avg_iou+iou
    #     sys.stdout.write('\rPgr:'+str(prg_counter/dataset_len*100*batch_size)+'%' ' L:'+ str(loss.item()))
    #     sys.stdout.write(' IoU:' +str(iou)+' pob:'+str(conf)+ ' nob:'+str(no_obj_conf))
    #     sys.stdout.write(' PCls:' +str(pos_class)+' ncls:'+str(neg_class))
    #     sys.stdout.flush()
    mAP=tester.get_map(model,confidence=0.01,iou_threshold=0.5)
    writer.add_scalar('Loss/train', total_loss/train_counter, e)
    writer.add_scalar('AIoU/train', avg_iou/train_counter, e)
    writer.add_scalar('PConf/train', avg_conf/train_counter, e)
    writer.add_scalar('NConf/train', avg_no_conf/train_counter, e)
    writer.add_scalar('PClass/train', avg_pos/train_counter, e)
    writer.add_scalar('NClass/train', avg_neg/train_counter, e)
    writer.add_scalar('mAP/valid', mAP, e)
    
    if mAP>mAP_max:
        torch.save(model.state_dict(),PATH+hyperparameters['path']+'_best.pth')
        mAP_max=mAP
    else:
        torch.save(model.state_dict(),PATH+hyperparameters['path']+'_last.pth')
        mAP_max=mAP
    scheduler.step()
    print('\ntotal number of misses is ' + str(misses))
    print('\n total average loss is '+str(total_loss/train_counter))
    print('\n total average iou is '+str(avg_iou/train_counter))