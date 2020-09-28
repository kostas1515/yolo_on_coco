import torch
import pandas as pd
import numpy as np
import cv2
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *
import numba
from numba import jit




def coco80_to_coco91_class(label):
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    return x[int(label)]


def standard(tensor):
    return (tensor-tensor.mean(dim=0))/tensor.std(dim=0)

def normalize(t):
    const=2.50662827
    t=(1/(2*t.std(dim=0)*const))*torch.exp(-1/2*(standard(t)**2))
    return t

def kl_div(pred,gt):
    sigma=(pred.std(dim=0)/gt.std(dim=0))**2
    mu=(pred.mean(dim=0)-gt.mean(dim=0))**2    
    kl=1/2*(mu*(1/pred.std(dim=0)**2+1/gt.std(dim=0)**2)+sigma+1/sigma-2)
    
    return kl

def get_idf(gt,mask):
    corpus_length=sum(mask)
#     tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()
    tf=1
    idf=gt[:,5:].sum(axis=0).cuda()

    idf=torch.log(corpus_length/idf)
    
    classes=gt[:,5:].max(1)[1]
    tfidf=tf*idf[classes]
    tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf.unsqueeze(1)

def get_precomputed_idf(obj_idf,col_name):
    
    idf=np.array(obj_idf[col_name])
    idf=torch.tensor(idf,device='cuda',dtype=torch.float)
    idf=-torch.log(idf)
    
#     tfidf=torch.softmax(tfidf,dim=0)
    
    return idf

def get_location_weights(offset,mask):
    final=[]
    for sl in offset.tolist():
        final.append(''.join(str(sl)))
    weights=[]
    values, counts = np.unique(final, return_counts=True)
    counts=np.log(len(mask)/counts)
    for el in final:
        for k,v in enumerate(values):
            if v==el:
                weights.append(counts[k])
    return torch.tensor(weights,device='cuda')

def get_weights(gt,mask,obj_idf,col_name):
    '''
    this function takes the ground truth loc or scale of the object belonging in a batch 
    and returns the diff from the average, which is calculated from the whole dataset.
    '''
    classes=gt[:,5:].max(1)[1]
    if col_name=='area':
        var=torch.tensor(obj_idf[col_name]).cuda()
        dset_var=var[classes.tolist()]
        gt_var=gt[:,3]*gt[:,2]
    elif col_name=='xc':
        var=torch.tensor(obj_idf[col_name]).cuda()
        dset_var=var[classes.tolist()]
        gt_var=gt[:,0]
    elif col_name=='yc':
        var=torch.tensor(obj_idf[col_name]).cuda()
        dset_var=var[classes.tolist()]
        gt_var=gt[:,1]
    elif (col_name=='obj_freq')|(col_name=='img_freq'):
#         tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()
        tf=1
        idf=np.array(obj_idf[col_name][classes.tolist()])
        idf=torch.tensor(idf,device='cuda')
        idf=-torch.log(idf)
        tfidf=tf*idf
        return tfidf.unsqueeze(1)
    elif (col_name=='random'):
        return torch.rand(gt.shape[0]).unsqueeze(1).cuda()
    else:
        return torch.tensor(1.0)
    
    weights=torch.abs(dset_var-gt_var)
    
    return weights.unsqueeze(1)

def dic2tensor(targets,key):
    
    tensor=torch.cat([t[key] for t in targets],dim=0)
    
    return tensor

def get_progress_stats(true_pred,no_obj,iou_list,targets):
    '''
    this function takes the Tranformed true prediction and the IoU list between the true prediction and the Gt boxes.
    Based on the IoU list it will calculate the mean IoU, the mean Positive Classification,  the mean negative Classification,
    the mean Positive objectness and the mean negative objectness.
    INPUTS: True_pred = Tensor:[N,BBs,4+1+C]
            no_obj_conf= Tensor [K]
            Targets = List of DICTs(N elements-> Key:Tenor[M,x]) , where M is the number of objects for that N, x depends on key.
            IoU List= List(N elements->Tensors[M,BBs])
    Outputs:DICT:{floats: avg_pos_conf, avg_neg_conf, avg_pos_class, avg_neg_class, avg_iou}  
    '''

    resp_true_pred=[]
    best_iou=[]
    no_obj2=no_obj.clone().detach()
    labels=dic2tensor(targets,'labels')
    
    for i in range(len(iou_list)):
        best_iou_positions=iou_list[i].max(axis=1)[1]
        best_iou.append(iou_list[i].max(axis=1)[0])
        
        resp_true_pred.append(true_pred[i,:,:][best_iou_positions])
        
    resp_true_pred=torch.cat(resp_true_pred,dim=0)
    best_iou=torch.cat(best_iou,dim=0)
    
    
    nb_digits = resp_true_pred.shape[1]-5 # get the number of CLasses
    n_obj=resp_true_pred.shape[0]
    y = labels.view(-1,1)
    y_onehot = torch.BoolTensor(n_obj, nb_digits)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    
    pos_class=resp_true_pred[:,5:][y_onehot].mean().item()
    neg_class=resp_true_pred[:,5:][~y_onehot].mean().item()
    
    pos_conf=resp_true_pred[:,4].mean().item()
    avg_iou=best_iou.mean().item()
    
    neg_conf=torch.sigmoid(no_obj2).mean().item()
    
    return  {'iou':avg_iou,
             'pos_conf':pos_conf,
             'neg_conf':neg_conf,
             'pos_class':pos_class,
             'neg_class':neg_class}
    


def collate_fn(batch):
    pictures=[i[0] for i in batch]
    pictures=torch.cat(pictures, dim=0)
    
    targets=[i[1] for i in batch]
        
    return pictures,targets

def convert2_abs(bboxes,shape):
        
    (h,w,c)=shape

    bboxes[:,1]=bboxes[:,1]*w
    bboxes[:,2]=bboxes[:,2]*h
    bboxes[:,3]=bboxes[:,3]*w
    bboxes[:,4]=bboxes[:,4]*h
    bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2
    bboxes[:,2]=bboxes[:,2]-bboxes[:,4]/2
    bboxes[:,3]=bboxes[:,1]+bboxes[:,3]
    bboxes[:,4]=bboxes[:,2]+bboxes[:,4]
    return bboxes


def convert2_abs_xyxy(bboxes,shape,inp_dim=1):
        
    (h,w,c)=shape
    h=h/inp_dim
    w=w/inp_dim
    
    
    xmin=bboxes[:,0]*w
    ymin=bboxes[:,1]*h
    width=bboxes[:,2]*w
    height=bboxes[:,3]*h
    xmin=xmin-width/2
    ymin=ymin-height/2
    xmax=xmin+width
    ymax=ymin+height
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xmin, ymin, xmax, ymax)).T
    else:
        return np.stack((xmin, ymin, xmax, ymax)).T



def convert2_rel_xcycwh(bboxes,shape):
        
    (h,w,c)=shape
    
    xmin=bboxes[:,0]/w
    ymin=bboxes[:,1]/h
    xmax=bboxes[:,2]/w
    ymax=bboxes[:,3]/h
        
    xc = (xmax- xmin)/2 +xmin
    yc = (ymax- ymin)/2 +ymin

    width = (xmax- xmin)
    height = (ymax- ymin)
    
    if (type(bboxes) is torch.Tensor):
        return torch.stack((xc, yc, width, height)).T
    else:
        return np.stack((xc, yc, width, height)).T
        

def write_pred(imgname,pred_final,inp_dim,max_detections,coco_version):
    for i in range(len(pred_final)):
        if pred_final[i].nelement() != 0:
            coord=pred_final[i][:,:4].cpu().detach().numpy()/inp_dim
            conf=pred_final[i][:,4:5].cpu().detach().numpy()
            mat=np.hstack((conf,coord))

            classes=pred_final[i][:,5:].max(1)[1].cpu().detach().numpy()
            classes=np.array([classes]).T

            mat=np.hstack((classes,mat))
            mat=np.array(mat)

            df=pd.DataFrame(mat,index=None,columns=None)
            df[0]=df[0].apply(lambda x: int(x))
            df=df[:max_detections]
            
            df.to_csv('../detections/coco'+coco_version+'/'+imgname[i]['img_name'],sep=' ',header=False,index=None)
        else:
            filename=open('../detections/coco'+coco_version+'/'+imgname[i]['img_name'],'w')

def transform_to_COCO(img_meta,pred_final):
    annotations=[]
    for i in range(len(pred_final)):
        if pred_final[i].nelement() != 0:
            conf=pred_final[i][:,4].cpu().detach().numpy()
            coord=pred_final[i][:,:4].cpu().detach().numpy()
            classes=pred_final[i][:,5:].max(1)[1].cpu().detach().numpy()
            area=(pred_final[i][:,2]*pred_final[i][:,3]).cpu().detach().numpy()
            for j in range(coord.shape[0]):
                annotations.append({ "segmentation":[], "area":float(area[j]), "iscrowd":0,"image_id":int(img_meta[i]['img_name'].split('.')[0]),"bbox": coord[j].tolist(),"score":float(conf[j]),"category_id": coco80_to_coco91_class(int(classes[j]))})
    return annotations
            

def getBoundingBoxes(coco_version):
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = '../'
    folderGT = os.path.join(currentPath,'labels/coco/labels/val'+coco_version)
    os.chdir(folderGT)
    files = glob.glob("*.txt")
    files.sort()
    # Class representing bounding boxes (ground truths and detections)
    allBoundingBoxes = BoundingBoxes()
    # Read GT detections from txt file
    # Each line of the files in the groundtruths folder represents a ground truth bounding box (bounding boxes that a detector should detect)
    # Each value of each line is  "class_id, x, y, width, height" respectively
    # Class_id represents the class of the bounding box
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        nameOfImage = f.replace(".txt","")
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue
            splitLine = line.split(" ")
            idClass = splitLine[0] #class
            x = float(splitLine[1]) #confidence
            y = float(splitLine[2])
            w = float(splitLine[3])
            h = float(splitLine[4])
            bb = BoundingBox(nameOfImage,idClass,x,y,w,h,typeCoordinates=CoordinatesType.Relative, imgSize=(416,416), bbType=BBType.GroundTruth, format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    # Read detections
    folderDet = os.path.join(currentPath,'../../../detections/coco'+coco_version)
    os.chdir(folderDet)
    files = glob.glob("*.txt")
    files.sort()
    # Read detections from txt file
    # Each line of the files in the detections folder represents a detected bounding box.
    # Each value of each line is  "class_id, confidence, x, y, width, height" respectively
    # Class_id represents the class of the detected bounding box
    # Confidence represents the confidence (from 0 to 1) that this detection belongs to the class_id.
    # x, y represents the most top-left coordinates of the bounding box
    # x2, y2 represents the most bottom-right coordinates of the bounding box
    for f in files:
        # nameOfImage = f.replace("_det.txt","")
        nameOfImage = f.replace(".txt","")
        # Read detections from txt file
        fh1 = open(f, "r")
        for line in fh1:
            line = line.replace("\n","")
            if line.replace(' ','') == '':
                continue            
            splitLine = line.split(" ")
            idClass = splitLine[0] #class
            confidence = float(splitLine[1]) #confidence
            x = float(splitLine[2])
            y = float(splitLine[3])
            w = float(splitLine[4])
            h = float(splitLine[5])
            bb = BoundingBox(nameOfImage, idClass,x,y,w,h,typeCoordinates=CoordinatesType.Relative,imgSize=(416,416),bbType=BBType.Detected, classConfidence=confidence, format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)
        fh1.close()
    os.chdir('../../yolo')
    return allBoundingBoxes