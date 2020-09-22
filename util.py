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
        prediction[:,:,5:] = torch.softmax((prediction[:,:, 5 :]).clone(),dim=2)
    
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
    



###by ultranalytics
###https://github.com/ultralytics/yolov3/blob/master/utils/utils.py

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

def same_picture_mask(responsible_mask,mask):
    '''
    mask is a batc list containing the number of objects per image
    a single image can contain many objects, so we must create a mask to 
    ignore the infuence of other targets into the no-oobj mask
    '''
    k=0
    for i,count in enumerate(mask):
        same_image_mask=False
        for obj in range(count):
            same_image_mask=same_image_mask+responsible_mask[:,k+obj]
        for obj in range(count):
            responsible_mask[:,k+obj]=same_image_mask
        k=k+count
    return responsible_mask

def transpose_target(box):
    
    if torch.cuda.is_available():
        box=box.cuda()
    xc = box[:,:,0]
    yc = box[:,:,1]
    
    w = box[:,:,2]
    h = box[:,:,3]
    
    return torch.stack((xc, yc, w, h)).T

def correct_iou_mask(iou_mask,fall_into_mask):
    ''' this function corrects the iou when, iou max =0 
    in that case for that object it decides that all responsible bboxes (9)
    should be considered for optimisation
    it also corrects the mask list, so it will have same number of objects
    le_mask has the true in the index of max, aka where iou.max=0
    '''
    le_mask=(iou_mask.sum(axis=0)==iou_mask.sum(axis=0).max())
    s=fall_into_mask[le_mask,:].shape
    a=torch.randint(0, 9, (1,))
    indices=((fall_into_mask[le_mask,:]==True).nonzero())
    indices=indices[:,1].reshape(9,s[0])
    indices=indices[a.item(),:].unsqueeze(0)
    array=torch.arange(s[1])
    array=array.repeat(s[0],1).T.cuda()
    one_hot_target = (indices == array)
    iou_mask[:,le_mask]=one_hot_target
    return iou_mask
    
def get_fall_into_mask(targets,offset,strd,inp_dim):
    #multiply by inp_dim then devide by stride to get the relative grid size coordinates, floor the result to get the corresponding cell
    target_xc=targets[:,0:1]
    target_yc=targets[:,1:2]
    
    #if target is on the boundary reduce it by small amount to bring it inside the grid cell
    target_xc[target_xc==1]=target_xc[target_xc==1]-1E-5
    target_yc[target_yc==1]=target_yc[target_yc==1]-1E-5
    
    centered_x=torch.floor(target_xc*inp_dim/strd.squeeze())
    centered_y=torch.floor(target_yc*inp_dim/strd.squeeze())
    fall_into_mask=(centered_x==offset[:,:,0])&(centered_y==offset[:,:,1])
    
    return fall_into_mask

def build_tensors(raw_pred,true_pred,anchors,offset,stride,fall_into_mask,mask):
    k=0
    counter=0
    temp_shape=raw_pred.shape[2]
    responsible_raw_pred=torch.empty([sum(mask),9,temp_shape],device='cuda')
    responsible_true_pred=torch.empty([sum(mask),9,temp_shape],device='cuda')
    for i in mask:
        for j in range(i):
            responsible_raw_pred[k,:,:]=raw_pred[counter,fall_into_mask[k]]
            responsible_true_pred[k,:,:]=true_pred[counter,fall_into_mask[k]]
            k=k+1
        counter=counter+1
    anchors=anchors[fall_into_mask].reshape(sum(mask),9,anchors.shape[2])
    offset=offset[fall_into_mask].reshape(sum(mask),9,offset.shape[2])
    stride=stride[fall_into_mask].reshape(sum(mask),9,stride.shape[2])
    return responsible_raw_pred,responsible_true_pred,anchors,offset,stride

def get_iou_mask(targets,responsible_true_pred,inp_dim,hyperparameters):
    
    iou_type=hyperparameters['iou_type']
    
    targets2=get_abs_coord(targets[:,:4]*inp_dim).unsqueeze(0)#inplace operations DANGER
    new_pred=get_abs_coord(responsible_true_pred)
    iou=bbox_iou(new_pred,targets2,iou_type,CUDA=True)
    iou_mask=iou.T==(iou.T.max(dim=1)[0].unsqueeze(1))
    
    iou_max,iou_max_index=(iou_mask.sum(axis=1).max(axis=0))
    while iou_max>1:
        iou_mask[iou_max_index][torch.randint(0,9,[1])]=False
        iou_mask[iou_max_index]=~iou_mask[iou_max_index]
        iou_max,iou_max_index=(iou_mask.sum(axis=1).max(axis=0))
    return iou,iou_mask

def get_noobj(true_pred,targets,fall_into_mask,mask,hyperparameters,inp_dim):
    prev=0
    counter=0
    iou_ignore_thresh=hyperparameters['iou_ignore_thresh']
    iou_type=hyperparameters['iou_type']
    no_obj_mask=[]
    for i in mask:
        combined_fall_into_mask=fall_into_mask[prev:prev+i,:].sum(axis=0,dtype=torch.bool) 
#         noobj=true_pred[counter,~combined_fall_into_mask,4:]
        noobj=true_pred[counter,~combined_fall_into_mask,4]
        abs_box=get_abs_coord(true_pred[counter,~combined_fall_into_mask,:4])
        abs_box=abs_box.unsqueeze(1)
        targets2=get_abs_coord(targets[:,:4]*inp_dim).unsqueeze(0)
        ignore_iou=bbox_iou(abs_box,targets2[:,prev:i+prev,:],iou_type,CUDA=True)
        ignore_iou_mask=(ignore_iou>iou_ignore_thresh).sum(axis=1,dtype=torch.bool)
        prev=i+prev
        no_obj_mask.append((~combined_fall_into_mask,~ignore_iou_mask))
#         no_obj_iou.append(1-ignore_iou.max(axis=1)[0])
#         no_obj.append(noobj)
#         print(ignore_iou.shape)
        counter=counter+1
    return no_obj_mask
    
def get_responsible_masks(transformed_output,targets,offset,strd,mask,inp_dim,hyperparameters):
    '''
    this function takes the transformed_output and
    the target box in respect to the resized image size
    and returns a mask which can be applied to select the 
    best raw input,anchors and cx_cy_offset
    and the noobj_mask for the negatives
    targets is a list
    '''
    #first transpose the centered normalised target coords
    ignore_threshold=hyperparameters['iou_ignore_thresh']
    iou_type=hyperparameters['iou_type']
    
    centered_target=transpose_target(targets)[:,:,0:2]
    #multiply by inp_dim then devide by stride to get the relative grid size coordinates, floor the result to get the corresponding cell
    centered_target=torch.floor(centered_target*inp_dim/strd)
    
    #create a mask to find where the gt falls into which gridcell in the grid coordinate system
    fall_into_mask=centered_target==offset
    fall_into_mask=fall_into_mask[:,:,0]&fall_into_mask[:,:,1]
#     fall_into_mask= ~fall_into_mask
    #create a copy of the transformed output
    best_bboxes=transformed_output.clone()
    #apply reverse mask to copy in order to zero all other bbox locations
    best_bboxes[~fall_into_mask]=0   
    #transform the copy to xmin,xmax,ymin,ymax
    best_responsible_coord=get_abs_coord(best_bboxes)
    targets=transpose_target(get_abs_coord(targets))*inp_dim
    #calculate best iou and mask
    responsible_iou=bbox_iou(best_responsible_coord,targets,iou_type,CUDA=True)
    responsible_iou[responsible_iou.ne(responsible_iou)] = 0
    responsible_mask=responsible_iou.max(dim=0)[0] == responsible_iou
    
    if(responsible_mask.sum()>sum(mask)):
        responsible_mask=correct_iou_mask(responsible_mask,fall_into_mask)
        
    abs_coord=get_abs_coord(transformed_output)
    iou=bbox_iou(abs_coord,targets,iou_type,CUDA=True)
    iou[iou.ne(iou)] = 0
    ignore_mask=ignore_threshold<=iou
    noobj_mask=~same_picture_mask(responsible_mask.clone()|ignore_mask,mask)
#     print(targets)
#     print(abs_coord.shape)
#     print(noobj_mask.shape)
#     print(noobj_mask.sum(dim=0))
#     print(abs_coord[noobj_mask].shape)
    
    return responsible_mask,noobj_mask

    
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

def yolo_loss(pred,gt,noobj_box,mask,anchors,offset,strd,inp_dim,hyperparameters):
    '''
    the targets correspon to single image,
    multiple targets can appear in the same image
    target has the size [objects,(tx,ty,tw.th,Confidence=1,class_i)]
    output should have the size [bboxes,(tx,ty,tw.th,Confidence,class_i)]
    inp_dim is the widht and height of the image specified in yolov3.cfg
    '''

    #box size has to be torch.Size([1, grid*grid*anchors, 6])
#     box0=output[:,:,:].squeeze(-3)# this removes the first dimension, maybe will have to change
    
    #box0[box0.ne(box0)] = 0 # this substitute all nan with 0
    xy_loss=0
    wh_loss=0
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
    
    
#     if hyperparameters['tfidf']==True:
#         if isinstance(hyperparameters['idf_weights'], pd.DataFrame):
#             class_weights=helper.get_weights(gt,mask,hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][0])
#             scale_weights=helper.get_weights(gt,mask,hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][1])
#             x_weights=helper.get_weights(gt,mask,hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][2]) 
#             y_weights=helper.get_weights(gt,mask,hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][3])
#             if(hyperparameters['tfidf_col_names'][4]=='softmax'):
#                 class_weights=torch.softmax(class_weights,dim=0)
#                 scale_weights=torch.softmax(scale_weights,dim=0)
#                 x_weights=torch.softmax(x_weights,dim=0)
#                 y_weights=torch.softmax(y_weights,dim=0)
#         else:#below code NEEDS FIXING
#             class_weights=helper.get_precomputed_idf(hyperparameters['idf_weights'],col_name=hyperparameters['tfidf_col_names'][0])
#             print(class_weights)
#             scale_weights=1
#             x_weights=1
#             y_weights=1
#     else:
#         class_weights=1
#         scale_weights=1
#         x_weights=1
#         y_weights=1

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
        gt[:,0:4]=gt[:,0:4]*inp_dim
        gt[:,0:4]=transform_groundtruth(gt,anchors,offset,strd)
        xy_loss=nn.MSELoss(reduction=hyperparameters['reduction'])
        xy_coord_loss=xy_loss(pred[:,0:2],gt[:,0:2])
        wh_loss=nn.MSELoss(reduction=hyperparameters['reduction'])
        wh_coord_loss=wh_loss(pred[:,2:4],gt[:,2:4])
    else:
        pred=transform(pred.unsqueeze(0),anchors.unsqueeze(0),offset.unsqueeze(0),strd.unsqueeze(0),only_coord=True).squeeze(0)
        gt[:,0:4]=gt[:,0:4]*inp_dim
        iou=bbox_iou(get_abs_coord(pred[:,0:4].unsqueeze(0)),get_abs_coord(gt[:,0:4].unsqueeze(0)),iou_type)
        if hyperparameters['reduction']=='sum':
            iou_loss=(1-iou).sum()
        else:
            iou_loss=(1-iou).mean()
       
    
    bce_class=nn.CrossEntropyLoss(reduction=hyperparameters['reduction'],weight=class_weights)
    class_loss=bce_class(pred[:,5:],gt[:,5:].max(axis=1)[1])

    
    bce_obj=csloss.FocalLoss(alpha=alpha,gamma=gamma,logits=True,reduction=hyperparameters['reduction'],pos_weight=loc_weights)
    confidence_loss=(bce_obj(pred[:,4],gt[:,4]))
    
    bce_noobj=csloss.FocalLoss(alpha=1-alpha,gamma=gamma,logits=True,reduction=hyperparameters['reduction'])
    no_obj_conf_loss=bce_noobj(noobj_box,torch.zeros(noobj_box.shape).cuda())
    
#     print(gt.shape[0])
#     print('iou_loss is:',iou_loss)
#     print('xy_loss is:',xy_coord_loss)
#     print('wh_coord_loss is:',wh_coord_loss)
#     print('confidence_loss is:',confidence_loss)
#     print('no_obj_conf_loss is:',no_obj_conf_loss)
#     print('class_loss is:',class_loss)

    total_loss=lcoord*(xy_coord_loss+wh_coord_loss+iou_loss)+confidence_loss+lno_obj*no_obj_conf_loss+class_loss
    
    if hyperparameters['reduction']=='sum':
        total_loss=total_loss/sum(mask)
    
    return total_loss

