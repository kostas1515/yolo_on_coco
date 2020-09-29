from __future__ import division
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2
import helper as helper
import custom_loss as csloss
import pandas as pd
import math
from scipy.stats import entropy
# from gluoncv import loss as gloss
   
def transform(prediction,anchors,x_y_offset,stride,CUDA = True,only_coord=False):
    '''
    This function takes the raw predicted output from yolo last layer in the correct
    '[batch_size,3*grid*grid,4+1+class_num] * grid_scale' size and transforms it into the real world coordinates
    Inputs: raw prediction, xy_offset, anchors, stride
    Output: real world prediction
    '''
    #Sigmoid the  centre_X, centre_Y.
    prediction[:,:,0] = torch.sigmoid(prediction[:,:,0])
    prediction[:,:,1] = torch.sigmoid(prediction[:,:,1])
    
    #Add the center offsets
    prediction[:,:,:2] += x_y_offset
    
    prediction[:,:,:2] = prediction[:,:,:2]*(stride)
    #log space transform height and the width
    prediction[:,:,2:4] = torch.exp(prediction[:,:,2:4]).clamp(max=1E4)*anchors*stride
    
    if(only_coord==False):
    #Sigmoid object confidencce
        prediction[:,:,4] = torch.sigmoid(prediction[:,:,4])
#         prediction[:,:,5: 5 + num_classes] = torch.sigmoid(prediction[:,:, 5 : 5 + num_classes])
        prediction[:,:,5:] = torch.softmax((prediction[:,:, 5 :]),dim=2)
    
    return prediction

def predict(prediction, inp_dim, anchors, num_classes, CUDA = True):
    '''
    this function reorders 4 coordinates tx,ty,tw,th as well as confidence and class probabilities
    then it sigmoids the confidence and the class probabilites
    Inputs: raw predictions from yolo last layer
    Outputs: pred: raw coordinate prediction, sigmoided confidence and class probabilities
    size of pred= [batch_size,3*grid*grid,4+1+class_num] in 3 different scales: grid, 2*gird,4*grid concatenated
    it also return stride, anchors and xy_offset in the same format to use later to transform raw output
    in the real world coordinates
    '''
    batch_size = prediction.size(0)
    stride =  inp_dim // prediction.size(2)
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
        
    
    return prediction



def get_utillities(stride,inp_dim, anchors, num_classes):
    '''
    this function reorders 4 coordinates tx,ty,tw,th as well as confidence and class probabilities
    then it sigmoids the confidence and the class probabilites
    Inputs: raw predictions from yolo last layer
    Outputs: pred: raw coordinate prediction, sigmoided confidence and class probabilities
    size of pred= [batch_size,3*grid*grid,4+1+class_num] in 3 different scales: grid, 2*gird,4*grid concatenated
    it also return stride, anchors and xy_offset in the same format to use later to transform raw output
    in the real world coordinates
    '''
    grid_size = inp_dim // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
    
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    
    #Add the center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1,1)
    y_offset = torch.FloatTensor(b).view(-1,1)


    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1,num_anchors).view(-1,2).unsqueeze(0)
    
    
    anchors = torch.FloatTensor(anchors)

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)
    
    strd=torch.ones(1,anchors.shape[1],1)*stride
    
    return anchors,x_y_offset,strd
    



##by ultranalytics
##https://github.com/ultralytics/yolov3/blob/master/utils/utils.py

def bbox_iou(box1, box2,iou_type,CUDA=True):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    GIoU, DIoU, CIoU=iou_type
    
    if CUDA:
        box2 = box2.cuda()
        box1 = box1.cuda()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,:,0], box1[:,:,1], box1[:,:,2], box1[:,:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,:,0], box2[:,:,1], box2[:,:,2], box2[:,:,3]

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = (w1 * h1 + 1e-16) + w2 * h2 - inter

    iou = inter / union  # iou
    if GIoU or DIoU or CIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + 1e-16  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # convex diagonal squared
            c2 = cw ** 2 + ch ** 2 + 1e-16
            # centerpoint distance squared
            rho2 = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2)) ** 2 / 4 + ((b2_y1 + b2_y2) - (b1_y1 + b1_y2)) ** 2 / 4
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU

    return iou


def get_iou_list(true_pred, targets,hyperparameters,inp_dim):
    '''
    Inputs: True_predictions:tensor[M,B,4+1+C], targets:List(Dict{boxes:tensor[N,4]}),hyperparameters['iou_type'], inp_dim : int num
    Outputs: List[M elements:tensor[N,B] iou_value of N->Bs]
    '''
    xyxy_true_pred=get_abs_coord(true_pred)
    xyxy_true_pred=xyxy_true_pred.transpose(0,1)
    iou_type=hyperparameters['iou_type']
#     iou_list=[]
#     for i in range(len(targets)):
#         gt_boxes=targets[i]['boxes']
#         gt_boxes=util.get_abs_coord(gt_boxes*inp_dim)
#         iou=bbox_iou(gt_boxes.unsqueeze(1), xyxy_true_pred[i,:,:].unsqueeze(0),iou_type).max(axis=1)
    
    iou_list=[bbox_iou(get_abs_coord(targets[i]['boxes']*inp_dim).unsqueeze(1), xyxy_true_pred[i,:,:].unsqueeze(0),iou_type) for i in range(len(targets))]
    
    return iou_list


def build_tensors(out,iou_list,pw_ph,cx_cy,stride,hyperparameters):
    resp_raw_pred=[]
    resp_cx_cy=[]
    resp_pw_ph=[]
    resp_stride=[]
    no_obj=[]
    
    iou_ignore_thresh=hyperparameters['iou_ignore_thresh']
    
    for i in range(len(iou_list)):
        best_iou_positions=iou_list[i].max(axis=1)[1]
        
        resp_raw_pred.append(out[i,:,:][best_iou_positions])
        resp_cx_cy.append(cx_cy[0,:,:][best_iou_positions])
        resp_pw_ph.append(pw_ph[0,:,:][best_iou_positions])
        resp_stride.append(stride[0,:,:][best_iou_positions])
        no_obj_mask=(iou_list[i]<iou_ignore_thresh).prod(axis=0)
        no_obj_mask[best_iou_positions]=0
        no_obj.append(out[i,no_obj_mask==1,4])

    resp_raw_pred=torch.cat(resp_raw_pred,dim=0)
    resp_cx_cy=torch.cat(resp_cx_cy,dim=0)
    resp_pw_ph=torch.cat(resp_pw_ph,dim=0)
    resp_stride=torch.cat(resp_stride,dim=0)
    no_obj=torch.cat(no_obj,dim=0)
    
    


    return resp_raw_pred,resp_cx_cy,resp_pw_ph,resp_stride,no_obj

    
def get_wasserstein_matrices(out,iou_list,inp_dim):
    
    w_d={}
    true_pred=torch.sigmoid(out[:,:,4:5])
    min_dim=int(inp_dim//32)
    tt=[]
    dd=[]
    for j in range(len(iou_list)):
        distribution=(iou_list[j].max(axis=0))[0]
        distribution[distribution<0]=0
        prev=0
        for i in range(3):
            slce=min_dim*(2**i)

            ind=slce*slce*3
            ba=true_pred[j,prev:prev+ind:3,0].reshape(slce,slce)
            bb=true_pred[j,prev+1:prev+ind:3,0].reshape(slce,slce)
            bc=true_pred[j,prev+2:prev+ind:3,0].reshape(slce,slce)
            t=torch.stack((ba,bb,bc)).cuda()
            tt.append(t)
            
            ba=distribution[prev:prev+ind:3].reshape(slce,slce)
            bb=distribution[prev+1:prev+ind:3].reshape(slce,slce)
            bc=distribution[prev+2:prev+ind:3].reshape(slce,slce)
            d=torch.stack((ba,bb,bc)).cuda()
            dd.append(d)
            prev=ind
    w_d={'dd':dd,'tt':tt}
    
    return w_d
    
def get_abs_coord(box):
    # yolo predicts center coordinates
    if torch.cuda.is_available():
        box=box.cuda()
    if (len(box.shape)==3):
        x1 = (box[:,:,0] - box[:,:,2]/2) 
        y1 = (box[:,:,1] - box[:,:,3]/2) 
        x2 = (box[:,:,0] + box[:,:,2]/2) 
        y2 = (box[:,:,1] + box[:,:,3]/2)
    else:
        x1 = (box[:,0] - box[:,2]/2) 
        y1 = (box[:,1] - box[:,3]/2) 
        x2 = (box[:,0] + box[:,2]/2) 
        y2 = (box[:,1] + box[:,3]/2)
    return torch.stack((x1, y1, x2, y2)).T

def xyxy_to_xywh(box):
    if torch.cuda.is_available():
        box=box.cuda()
    if (len(box.shape)==3):
        xc = (box[:,:,2]- box[:,:,0])/2 +box[:,:,0]
        yc = (box[:,:,3]- box[:,:,1])/2 +box[:,:,1]
        w = (box[:,:,2]- box[:,:,0])
        h = (box[:,:,3]- box[:,:,1])
    else:
        xc = (box[:,2]- box[:,0])/2 +box[:,0]
        yc = (box[:,3]- box[:,1])/2 +box[:,1]
        w = (box[:,2]- box[:,0])
        h = (box[:,3]- box[:,1])
    
    return torch.stack((xc, yc, w, h)).T

def transpose_target(box):
    
    if torch.cuda.is_available():
        box=box.cuda()
    xc = box[:,:,0]
    yc = box[:,:,1]
    
    w = box[:,:,2]
    h = box[:,:,3]
    
    return torch.stack((xc, yc, w, h)).T



    
def transform_groundtruth(target,anchors,cx_cy,strd):
    '''
    this function takes the target real coordinates and transfroms them into grid cell coordinates
    returns the groundtruth to use for optimisation step
    consider using sigmoid to prediction, insted of inversing groundtruth
    '''
    target[:,0:4]=target[:,0:4]/strd
    target[:,0:2]=target[:,0:2]-cx_cy
#     target[:,0:2][target[:,0:2]==0] =1E-5
#     target[:,0:2]=torch.log(target[:,0:2]/(1-target[:,0:2])).clamp(min=-10, max=10)
    target[:,2:4]=torch.log(target[:,2:4]/anchors)
    
    return target[:,0:4]




def yolo_loss(pred,targets,noobj_box,anchors,offset,strd,inp_dim,hyperparameters):
    '''
    the targets correspon to single image,
    multiple targets can appear in the same image
    target has the size [objects,(tx,ty,tw.th,Confidence=1,class_i)]
    output should have the size [bboxes,(tx,ty,tw.th,Confidence,class_i)]
    inp_dim is the widht and height of the image specified in yolov3.cfg
    '''
    gt_boxes=helper.dic2tensor(targets,'boxes').cuda()
    gt_labels=helper.dic2tensor(targets,'labels').cuda()
    #box size has to be torch.Size([1, grid*grid*anchors, 6])
#     box0=output[:,:,:].squeeze(-3)# this removes the first dimension, maybe will have to change
    
    #box0[box0.ne(box0)] = 0 # this substitute all nan with 0
    iou_loss=0
    xy_coord_loss=0
    wh_coord_loss=0
    class_loss=0
    confidence_loss=0
    total_loss=0
    no_obj_conf_loss=0
    no_obj_counter=0
    lcoord=hyperparameters['lcoord']
    lno_obj=hyperparameters['lno_obj']
    gamma=hyperparameters['gamma']
    alpha=hyperparameters['alpha']
    iou_type=hyperparameters['iou_type']
    

    if hyperparameters['tfidf']==True:
        if isinstance(hyperparameters['idf_weights'], pd.DataFrame):
            class_weights=helper.get_precomputed_idf(hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][0])
            if(hyperparameters['tfidf_col_names'][4]=='softmax'):
                class_weights=torch.softmax(class_weights,dim=0)
            elif(hyperparameters['tfidf_col_names'][4]=='minmax'):
                class_weights_std= (class_weights - class_weights.min(axis=0)[0]) / (class_weights.max(axis=0)[0] - class_weights.min(axis=0)[0])
                class_weights = class_weights_std +0.1
#             print(class_weights)
            scale_weights=1
            x_weights=1
            y_weights=1
    else:
        class_weights=None
        
    if(hyperparameters['tfidf_col_names'][1]=='area'):
        loc_weights=helper.get_location_weights(offset,mask)
    else:
        loc_weights=None
        
    if(iou_type==(0,0,0)):#this means normal training with mse
        pred[:,0] = torch.sigmoid(pred[:,0])
        pred[:,1]= torch.sigmoid(pred[:,1])
        gt_boxes[:,0:4]=gt_boxes[:,0:4]*inp_dim
        gt_boxes[:,0:4]=transform_groundtruth(gt_boxes,anchors,offset,strd)
        xy_loss=nn.MSELoss(reduction=hyperparameters['reduction'])
        xy_coord_loss=xy_loss(pred[:,0:2],gt_boxes[:,0:2])
        wh_loss=nn.MSELoss(reduction=hyperparameters['reduction'])
        wh_coord_loss=wh_loss(pred[:,2:4],gt_boxes[:,2:4])
    else:
        pred=transform(pred.unsqueeze(0),anchors.unsqueeze(0),offset.unsqueeze(0),strd.unsqueeze(0),only_coord=True).squeeze(0)
        gt_boxes[:,0:4]=gt_boxes[:,0:4]*inp_dim
        iou=bbox_iou(get_abs_coord(pred[:,0:4].unsqueeze(0)),get_abs_coord(gt_boxes[:,0:4].unsqueeze(0)),iou_type)
        if hyperparameters['reduction']=='sum':
            iou_loss=(1-iou).sum()
        else:
            iou_loss=(1-iou).mean()
       
    
    bce_class=nn.CrossEntropyLoss(reduction=hyperparameters['reduction'],weight=class_weights)
    class_loss=bce_class(pred[:,5:],gt_labels)

    if hyperparameters['wasserstein']==False:
        bce_obj=csloss.FocalLoss(alpha=alpha,gamma=gamma,logits=True,reduction=hyperparameters['reduction'],pos_weight=loc_weights)
        confidence_loss=(bce_obj(pred[:,4],torch.ones(pred[:,4].shape).cuda()))

        bce_noobj=csloss.FocalLoss(alpha=1-alpha,gamma=gamma,logits=True,reduction=hyperparameters['reduction'])
        no_obj_conf_loss=bce_noobj(noobj_box,torch.zeros(noobj_box.shape).cuda())
    else:    
        sinkhorn = csloss.SinkhornDistance(eps=0.1, max_iter=100,reduction='sum')
        no_obj_conf_loss=0
        for i in range(len(noobj_box['tt'])):
            dist, P, C = sinkhorn(noobj_box['dd'][i], noobj_box['tt'][i])
            no_obj_conf_loss+=dist
        
#     print(total_dist)
#     print(gt_labels.shape[0])
#     print('iou_loss is:',iou_loss)
#     print('xy_loss is:',xy_coord_loss)
#     print('wh_coord_loss is:',wh_coord_loss)
#     print('confidence_loss is:',confidence_loss)
#     print('no_obj_conf_loss is:',no_obj_conf_loss)
#     print('class_loss is:',class_loss)

    total_loss=lcoord*(xy_coord_loss+wh_coord_loss+iou_loss)+confidence_loss+lno_obj*no_obj_conf_loss+class_loss
#     total_loss=lcoord*(xy_coord_loss+wh_coord_loss+iou_loss)+class_loss+total_dist
    
    if hyperparameters['reduction']=='sum':
        total_loss=total_loss/gt_labels.shape[0]
    
    return total_loss

