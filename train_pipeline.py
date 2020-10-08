from dataset import *

import torch
import torch.nn as nn
import torch.optim as optim
import test
import helper 
from torch.utils.tensorboard import SummaryWriter
import yolo_function as yolo_function
import init_model


hyperparameters={'lr': 0.0001, 
                 'epochs': 90,
                 'resume_from':0,
                 'coco_version': '2014', #can be either '2014' or '2017'
                 'batch_size': 16,
                 'weight_decay': 0.0039,
                 'momentum': 0.76, 
                 'optimizer': 'sgd', 
                 'alpha': 0.8808, 
                 'gamma': 1.343, 
                 'lcoord': 3.52,
                 'lno_obj': 1.53,
                 'iou_type': (0, 0, 0),
                 'iou_ignore_thresh': 0.4194, 
                 'tfidf': True, 
                 'idf_weights': True, 
                 'tfidf_col_names': ['img_freq', 'none', 'none', 'none', 'no_softmax'],
                 'wasserstein':False,
                 'inf_confidence':0.01,
                 'inf_iou_threshold':0.5,
                 'augment': 1, 
                 'workers': 4,
                 'pretrained':False,
                 'path': 'yolo2014_with_bayes', 
                 'reduction': 'sum'}

mode={'bayes_opt':True,
      'multi_scale':True,
      'debugging':False,
      'show_output':True,
      'multi_gpu':False,
      'show_temp_summary':True,
      'save_summary': False
     }

if(mode['show_temp_summary']==True):
    writer = SummaryWriter('../results/test_vis/')
if(mode['save_summary']==True):
    writer = SummaryWriter('../results/'+hyperparameters['path'])    
mAP_best=0
coco_version=hyperparameters['coco_version']


for i in range(hyperparameters['epochs']):
    
    model,optimizer,hyperparameters=init_model.init_model(hyperparameters,mode)

    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
    else:
        inp_dim=model.inp_dim


    if hyperparameters['augment']>0:
        train_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([Augment(hyperparameters['augment']),ResizeToTensor(inp_dim)]))
    else:
        train_dataset=Coco(partition='train',coco_version=coco_version,transform=transforms.Compose([ResizeToTensor(inp_dim)]))

    batch_size=hyperparameters['batch_size']


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True,collate_fn=helper.collate_fn, num_workers=hyperparameters['workers'])



    outcome=yolo_function.train_one_epoch(model,optimizer,train_dataloader,hyperparameters,mode)
    mAP=test.evaluate(model, device,coco_version,confidence=hyperparameters['inf_confidence'],iou_threshold=hyperparameters['inf_iou_threshold'])
    if(mode['save_summary']==True):
        writer.add_scalar('Loss/train', outcome['avg_loss'], hyperparameters['resume_from'])
        writer.add_scalar('AIoU/train', outcome['avg_iou'], hyperparameters['resume_from'])
        writer.add_scalar('PConf/train', outcome['avg_conf'], hyperparameters['resume_from'])
        writer.add_scalar('NConf/train', outcome['avg_no_conf'], hyperparameters['resume_from'])
        writer.add_scalar('PClass/train', outcome['avg_pos'], hyperparameters['resume_from'])
        writer.add_scalar('NClass/train', outcome['avg_neg'], hyperparameters['resume_from'])
        
        writer.add_scalar('mAP/valid', mAP, hyperparameters['resume_from'])
        
        hyperparameters['resume_from']=hyperparameters['resume_from']+1

    torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['avg_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'epoch':hyperparameters['resume_from'],
            'mAP': mAP,
            'hyperparameters': hyperparameters
            }, PATH+hyperparameters['path']+'.tar')

        
    if mAP>mAP_best:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': outcome['avg_loss'],
            'avg_iou': outcome['avg_iou'],
            'avg_pos': outcome['avg_pos'],
            'avg_neg':outcome['avg_neg'],
            'avg_conf': outcome['avg_conf'],
            'avg_no_conf': outcome['avg_no_conf'],
            'epoch':hyperparameters['resume_from'],
            'mAP': mAP,
            'hyperparameters': hyperparameters
            }, PATH+hyperparameters['path']+'_best.tar')
        mAP_best=mAP
        
    