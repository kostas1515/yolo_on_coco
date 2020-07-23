import torch
import pandas as pd
import numpy as np
import cv2
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

def get_responsible_boxes(true_pred,fall_into_mask,mask):  
    responsible_boxes=torch.empty([sum(mask),9,true_pred.shape[2]],device='cuda')
    counter=0
    for i in mask:
        for j in range(i):
            responsible_boxes[k,:,:]=true_pred[counter,fall_into_mask[k]]
            k=k+1
        counter=counter+1

def coco80_to_coco91_class(label):
    x= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    
    return x[int(label)]

def my_collate(batch):
    write=0
    boxes=[]
    image_meta=[]
    pictures=[]
    for el in filter(None,batch):
        if write==0:
            pictures=[img.unsqueeze(-4) for img in el['images']]
            pictures=torch.cat(pictures, dim=0)
            write=1
        else:
            pics=[img.unsqueeze(-4) for img in el['images']]
            pics=torch.cat(pics, dim=0)
            pictures=torch.cat((pictures,pics),0)
        boxes=boxes+el['boxes']
        image_meta.append({'img_name':el['img_name'],'img_size':el['img_size']})
    return pictures,boxes,image_meta


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

def get_precomputed_idf(gt,mask,obj_idf,col_name):
    
    tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()

    classes=gt[:,5:].max(1)[1]
    
    idf=np.array(obj_idf[col_name][classes.tolist()])
    idf=torch.tensor(idf,device='cuda')
    idf=-torch.log(idf)
    
    tfidf=tf*idf
#     tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf.unsqueeze(1)


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

def convert2_abs_xywh(bboxes,shape,inp_dim):
        
    (h,w,c)=shape
    h=h/inp_dim
    w=w/inp_dim
        
    bboxes[:,0]=bboxes[:,0]*w
    bboxes[:,1]=bboxes[:,1]*h
    bboxes[:,2]=bboxes[:,2]*w
    bboxes[:,3]=bboxes[:,3]*h
    bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2
    bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2
        
    return bboxes

def convert2_rel(bboxes,shape):
        
    (h,w,c)=shape
        
    bboxes[:,1]=bboxes[:,1]/w
    bboxes[:,2]=bboxes[:,2]/h
    bboxes[:,3]=bboxes[:,3]/w
    bboxes[:,4]=bboxes[:,4]/h
        
    bboxes[:,1] = (bboxes[:,3]- bboxes[:,1])/2 +bboxes[:,1]
    bboxes[:,2] = (bboxes[:,4]- bboxes[:,2])/2 +bboxes[:,2]

    bboxes[:,3] = 2*(bboxes[:,3]- bboxes[:,1])
    bboxes[:,4] = 2*(bboxes[:,4]- bboxes[:,2])
        
    return bboxes
        

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