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


def get_map(model,confidence,iou_threshold):
    
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
        
    model.eval()
    
    
    max_detections=None
    transformed_dataset=Coco(partition='val',
                                               transform=transforms.Compose([
                                                ResizeToTensor(inp_dim)
                                               ]))



    dataset_len=(len(transformed_dataset))
#     print('Length of dataset is '+ str(dataset_len)+'\n')
    batch_size=8

    dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                            shuffle=True,collate_fn=helper.my_collate, num_workers=2)


    for images,targets,img_name in dataloader:
        inp=images.cuda()
        raw_pred = model(inp, torch.cuda.is_available())
        targets,anchors,offset,strd,mask=helper.collapse_boxes(targets,pw_ph,cx_cy,stride)

        raw_pred=raw_pred.to(device='cuda')
        true_pred=util.transform(raw_pred.clone(),pw_ph,cx_cy,stride)
        
        sorted_pred=torch.sort(true_pred[:,:,4],descending=True)
        pred_mask=sorted_pred[0]>confidence
        indices=[(sorted_pred[1][e,:][pred_mask[e,:]]) for e in range(pred_mask.shape[0])]
        pred_final=[true_pred[i,indices[i],:] for i in range(len(indices))]

        pred_final_coord=[util.get_abs_coord(pred_final[i].unsqueeze(-2)) for i in range(len(pred_final))]

        indices=[nms_box.nms(pred_final_coord[i][0],pred_final[i][:,4],iou_threshold) for i in range(len(pred_final))]

        pred_final=[pred_final[i][indices[i],:] for i in range(len(pred_final))]

    #     pred_final[:,0:4]=pred_final[:,0:4]/inp_dim
        helper.write_pred(img_name,pred_final,inp_dim,max_detections)
        
    boundingboxes = helper.getBoundingBoxes()
        

    evaluator = Evaluator()


    metricsPerClass = evaluator.GetPascalVOCMetrics(boundingboxes, IOUThreshold=0.5)
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