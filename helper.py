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
    corpus_length=len(mask)
    tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()
    idf=torch.stack([gt[:i,5:].sum(axis=0) for i in mask],dim=0)
    idf[idf>1]=1
    idf=torch.log(corpus_length/idf.sum(axis=0))
    idf[idf== float('inf')] = 0
    classes=gt[:,5:].max(1)[1]
    
    tfidf=tf*idf[classes]
    tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf

def get_precomputed_idf(gt,mask,obj_idf):
    
    tf=torch.tensor([1/mask[i] for i in range(len(mask)) for j in range(mask[i])]).cuda()

    classes=gt[:,5:].max(1)[1]
    
    idf=np.array(obj_idf['obj_idf'][classes.tolist()])
    idf=torch.tensor(idf,device='cuda')
    idf=-torch.log(idf)
    
    tfidf=tf*idf
    tfidf=torch.softmax(tfidf,dim=0)
    
    return tfidf



def get_area_weights(gt):
    area=torch.abs(gt[:,2]*gt[:,3])
    weights=torch.softmax(torch.sqrt(area),dim=0).cuda()
    
    return weights

def clip_box(bbox, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `xc yc w h`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `xc yc w h` 
    
    """
    ar_=bbox[:,2]*bbox[:,3]
    
    x1=bbox[:,0]-bbox[:,2]/2
    y1=bbox[:,1]-bbox[:,3]/2
    x2=bbox[:,0]+bbox[:,2]/2
    y2=bbox[:,1]+bbox[:,3]/2
    
    x_min = torch.max(x1,torch.zeros(x1.shape))
    y_min = torch.max(y1,torch.zeros(y1.shape))
    x_max = torch.min(x2,torch.ones(x2.shape))
    y_max = torch.min(y2,torch.ones(y2.shape))
    
    in_area_=(x_max-x_min)*(y_max-y_min)
    
    delta_area = ((ar_ - in_area_)/ar_)
    
    mask = delta_area < (1 - alpha)

    
    bbox = bbox[mask,:]


    return bbox
  

def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image


def get_corners(bboxes):
    
    """Get corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `xc yc w h`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
        
    """
    width = bboxes[:,2]-bboxes[:,0]
    height = bboxes[:,3]-bboxes[:,1]
    
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = x1 + width
    y4 = y1 + height
    
    corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4),dim=1)
    
    return corners

def rotate_box(corners,angle,  cx, cy, h, w):
    
    """Rotate the bounding box.
    
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """
    corners = corners.reshape(-1,2)
    corners = torch.cat([corners, torch.ones((corners.shape[0],1))],dim=1)
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = ((h * sin) + (w * cos))
    nH = ((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += ((nW / 2) - cx)
    M[1, 2] += ((nH / 2) - cy)
    
    
    
    # Prepare the vector to be transformed
    M=torch.tensor(M,dtype=torch.float32)
    calculated = torch.mm(M,corners.T).T
    calculated = calculated.reshape(-1,8)

    return calculated


def get_enclosing_box(corners):
    """Get an enclosing box for ratated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:,[0,2,4,6]]
    y_ = corners[:,[1,3,5,7]]

    xmin = torch.min(x_,dim=1)[0].reshape(-1,1)
    ymin = torch.min(y_,dim=1)[0].reshape(-1,1)
    xmax = torch.max(x_,dim=1)[0].reshape(-1,1)
    ymax = torch.max(y_,dim=1)[0].reshape(-1,1)
    
    
    
#     w=xmax-xmin
#     h=ymax-ymin
#     xc=xmin + w/2
    
#     yc=(ymin + h/2 +ymax-h/2)/2
    
    final = torch.cat([xmin, ymin, xmax, ymax,corners[:,8:]],dim=1)
    
#     final = torch.cat([xc, yc, w, h,corners[:,8:]],dim=1)
    
    return final


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
        

def write_pred(imgname,pred_final,inp_dim,max_detections):
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