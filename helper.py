import torch
import pandas as pd
import numpy as np
from BoundingBox import BoundingBox
from BoundingBoxes import BoundingBoxes
from utils import *

def collapse_boxes(boxes,pw_ph,cx_cy,stride):
    write=0
    mask=[]
    for box in boxes:
        if write==0:
            targets=box
            anchors=torch.stack([pw_ph for p in range(box.shape[0])], dim=0)
            offset=torch.stack([cx_cy for p in range(box.shape[0])], dim=0)
            strd=torch.stack([stride for p in range(box.shape[0])], dim=0)
            write=1
        else:
            targets=torch.cat((targets,box),0)
            
            anchors=torch.cat((anchors,torch.stack([pw_ph for p in range(box.shape[0])], dim=0)),0)
            offset=torch.cat((offset,torch.stack([cx_cy for p in range(box.shape[0])], dim=0)),0)
            strd=torch.cat((strd,torch.stack([stride for p in range(box.shape[0])], dim=0)),0)
        mask.append(box.shape[0])
    return targets,anchors.squeeze(1),offset.squeeze(1),strd.squeeze(1),mask

def expand_predictions(predictions,mask):
    k=0
    write=0
    for i in mask:
        if write==0:
            new=torch.stack([predictions[k,:,:] for p in range(i)], dim=0)
            write=1
        else:
            new=torch.cat((new,torch.stack([predictions[k,:,:] for p in range(i)], dim=0)),0)
        k=k+1
    
    return new

def uncollapse(predictions,mask):
    k=0
    write=0
    for i in mask:
        if write==0:
            new=torch.stack([predictions[k,:,:]], dim=0)
            write=1
        else:
            new=torch.cat((new,torch.stack([predictions[k,:,:]], dim=0)),0)
        k=k+i
    
    return new


def my_collate(batch):
    write=0
    boxes=[]
    img_name=[]
    pictures=[]
    for el in filter(None,batch):
        if write==0:
            pictures=el['images'].unsqueeze(-4)
            write=1
        else:
            pictures=torch.cat((pictures,el['images'].unsqueeze(-4)),0)
        boxes.append(el['boxes'])
        img_name.append(el['img_name'])
    return pictures,boxes,img_name


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
    tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()
    idf=torch.log(corpus_length/gt[:,5:].sum(axis=0))
    idf[idf== float('inf')] = 0
    classes=gt[:,5:].max(1)[1]
    
    tfidf=tf*idf[classes]
    tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf,idf

def get_area_weights(gt):
    area=torch.abs(gt[:,2]*gt[:,3])
    weights=torch.softmax(torch.sqrt(area),dim=0).cuda()
    
    return weights
  
    

def write_pred(imgname,pred_final,inp_dim):
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
            
            df.to_csv('../detections/'+imgname[i],sep=' ',header=False,index=None)
        else:
            filename=open('../detections/'+imgname[i],'w')
            

def getBoundingBoxes():
    """Read txt files containing bounding boxes (ground truth and detections)."""
    allBoundingBoxes = BoundingBoxes()
    import glob
    import os
    # Read ground truths
    currentPath = '..'
    folderGT = os.path.join(currentPath,'labels/coco/labels/val2017')
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
    folderDet = os.path.join(currentPath,'../../../detections')
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
    return allBoundingBoxes