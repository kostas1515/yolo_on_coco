from dataset import *
import torch
import torch.optim as optim
import test 
import sys
import helper as helper
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import test
import init_model
import yolo_function as yolo_function


def bayesian_opt(w,m,g,a,lcoor,lno,iou_thresh,iou_type,bayes_opt=True):
    
    iou_type=int(round(iou_type))
    if(iou_type)==0:
        iou_type=(0,0,0)
    elif(iou_type==1):
        iou_type=(1,0,0)
    elif(iou_type==2):
        iou_type=(0,1,0)
    else:
        iou_type=(0,0,1) 
    
    hyperparameters={'lr': 0.0001, 
                     'epochs': 1,
                     'resume_from':0,
                     'coco_version': '2017', #can be either '2014' or '2017'
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
                     'inf_confidence':0.01,
                     'inf_iou_threshold':0.5,
                     'wasserstein':False,
                     'tfidf': True, 
                     'idf_weights': True, 
                     'tfidf_col_names': ['img_freq', 'none', 'none', 'none', 'no_softmax'],
                     'augment': 1, 
                     'workers': 4,
                     'pretrained':False,
                     'path': 'yolo2017_semiprtnd', 
                     'reduction': 'sum'}

    mode={'bayes_opt':bayes_opt,
          'multi_scale':False,
          'debugging':False,
          'show_output':False,
          'multi_gpu':True,
          'show_temp_summary':False,
          'save_summary': bayes_opt==False
         }

#     print(hyperparameters)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print('Using: ',device)

    model,optimizer,hyperparameters=init_model.init_model(hyperparameters,mode,show=False)

    if type(model) is nn.DataParallel:
        inp_dim=model.module.inp_dim
    else:
        inp_dim=model.inp_dim
    coco_version=hyperparameters['coco_version']

            
            
    if bayes_opt==True:
        tr_subset=0.1
        ts_subset=1
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
        if(mode['save_summary']==True):
            
            writer.add_scalar('Loss/train', outcome['avg_loss'], hyperparameters['resume_from'])
            writer.add_scalar('AIoU/train', outcome['avg_iou'], hyperparameters['resume_from'])
            writer.add_scalar('PConf/train', outcome['avg_conf'], hyperparameters['resume_from'])
            writer.add_scalar('NConf/train', outcome['avg_no_conf'], hyperparameters['resume_from'])
            writer.add_scalar('PClass/train', outcome['avg_pos'], hyperparameters['resume_from'])
            writer.add_scalar('NClass/train', outcome['avg_neg'], hyperparameters['resume_from'])
            writer.add_scalar('mAP/valid', mAP, hyperparameters['resume_from'])
            
            hyperparameters['resume_from']=hyperparameters['resume_from']+1
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
            'epoch':hyperparameters['resume_from']
            }, PATH+hyperparameters['path']+'.tar')

#             hyperparameters['resume_from']=checkpoint['epoch']+1
                
    return mAP