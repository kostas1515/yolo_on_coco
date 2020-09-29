from darknet import *
import darknet as dn
import util as util
import torch.optim as optim
import pandas as pd
import time
import sys
import timeit
from dataset import *
import torchvision.ops.boxes as nms_box
import helper as helper
from utils import *
from Evaluator import *
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
import coco_utils
import coco_eval



def get_map(model,confidence,iou_threshold,coco_version,subset=1):
    
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
    
    pw_ph=pw_ph.cuda()
    cx_cy=cx_cy.cuda()
    stride=stride.cuda()
    
    model.eval()
    subset=subset
    
    max_detections=100
    transformed_dataset=Coco(partition='val',coco_version=coco_version,subset=subset,
                                               transform=transforms.Compose([
                                                ResizeToTensor(inp_dim)
                                               ]))



    dataset_len=(len(transformed_dataset))
#     print('Length of dataset is '+ str(dataset_len)+'\n')
    batch_size=8

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=False,collate_fn=helper.my_collate, num_workers=4)


    for images,targets in dataloader:
        inp=images.cuda()
        raw_pred = model(inp, torch.cuda.is_available())
        true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)

        sorted_pred=torch.sort(true_pred[:,:,4]*(true_pred[:,:,5:].max(axis=2)[0]),descending=True)
        
        pred_mask=sorted_pred[0]>confidence
        indices=[(sorted_pred[1][e,:][pred_mask[e,:]]) for e in range(pred_mask.shape[0])]
        pred_final=[true_pred[i,indices[i],:] for i in range(len(indices))]

        pred_final_coord=[util.get_abs_coord(pred_final[i].unsqueeze(-2)) for i in range(len(pred_final))]

        indices=[nms_box.nms(pred_final_coord[i][0],pred_final[i][:,4],iou_threshold) for i in range(len(pred_final))]

        pred_final=[pred_final[i][indices[i],:] for i in range(len(pred_final))]

    #     pred_final[:,0:4]=pred_final[:,0:4]/inp_dim
        helper.write_pred(img_name,pred_final,inp_dim,max_detections,coco_version)
        
    boundingboxes = helper.getBoundingBoxes(coco_version)
        

    evaluator = Evaluator()


    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.75)
    # Loop through classes to obtain their metrics
    mAP=0
    counter=0
    for mc in metricsPerClass:
        # Get metric values per each class
        c = mc['class']
        precision = mc['precision']
        recall = mc['recall']
        average_precision = mc['AP']
        ipre = mc['interpolated precision']
        irec = mc['interpolated recall']
        # Print AP per class
        mAP=average_precision+mAP
#         print('%s: %f' % (c, average_precision))

#     print('map is:',mAP/80)
    return mAP/80



def evaluate(model, device,coco_version,confidence=0.01,iou_threshold=0.5,subset=1):
    # FIXME remove this and make paste_masks_in_image run on the GPU
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")

    device = device
    model.eval()

    
#     metric_logger = utils.MetricLogger(delimiter="  ")
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
    
    pw_ph=pw_ph.to(device)
    cx_cy=cx_cy.to(device)
    stride=stride.to(device)
    

    subset=subset
    

    
    transformed_dataset=Coco(partition='val',coco_version=coco_version,subset=subset,
                                               transform=transforms.Compose([
                                                ResizeToTensor(inp_dim)
                                               ]))
    
    dataloader = DataLoader(transformed_dataset, batch_size=8,
                            shuffle=False,collate_fn=helper.collate_fn, num_workers=4)
    
    coco = coco_utils.get_coco_api_from_dataset(transformed_dataset)
    iou_types = ["bbox"]
    coco_evaluator = coco_eval.CocoEvaluator(coco, iou_types)

    for images,targets in dataloader:
        images = images.to(device)
        
        targets2=[]
        for t in targets:
            dd={}
            for k, v in t.items():
                if(k!='img_size'):
                    dd[k]=v.to(device)
                else:
                    dd[k]=v
            targets2.append(dd)
            
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        targets=targets2
    
        with torch.no_grad():
            raw_pred = model(images, device)
        
        true_pred=util.transform(raw_pred.clone().detach(),pw_ph,cx_cy,stride)

        sorted_pred=torch.sort(true_pred[:,:,4]*(true_pred[:,:,5:].max(axis=2)[0]),descending=True)
        pred_mask=sorted_pred[0]>confidence
        indices=[(sorted_pred[1][e,:][pred_mask[e,:]]) for e in range(pred_mask.shape[0])]
        pred_final=[true_pred[i,indices[i],:] for i in range(len(indices))]
        
        pred_final_coord=[util.get_abs_coord(pred_final[i].unsqueeze(-2)) for i in range(len(pred_final))]
        
        indices=[nms_box.nms(pred_final_coord[i][0],pred_final[i][:,4],iou_threshold) for i in range(len(pred_final))]
        pred_final=[pred_final[i][indices[i],:] for i in range(len(pred_final))]
        
        
        abs_pred_final=[helper.convert2_abs_xyxy(pred_final[i],targets[i]['img_size'],inp_dim) for i in range(len(pred_final))]

        
        outputs=[dict() for i in range(len((abs_pred_final)))]
        for i,atrbs in enumerate(abs_pred_final):
            
            outputs[i]['boxes']=atrbs[:,:4]
            outputs[i]['scores']=pred_final[i][:,4]
            try:
                outputs[i]['labels']=pred_final[i][:,5:].max(axis=1)[1] +1 #could be empty
            except:

                outputs[i]['labels']=torch.tensor([])
            

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        coco_evaluator.update(res)
#         print('det is ',res)
        
    # gather the stats from all processes
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator.get_stats()[0]
