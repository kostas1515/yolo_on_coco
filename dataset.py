from __future__ import print_function, division
from torch.autograd import Variable
import numpy as np
import cv2
import os
import torch
import io
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random

class Coco(Dataset):

    def __init__(self, partition, transform=None):
        """
        Args:
            zip_file (string): Path to the zip file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.pointers=pd.read_csv('../pointers/'+partition+'2017.txt',names=['img'])
        self.pointers['box']=self.pointers['img'].apply(lambda x: x.split('.')[0]+'.txt')
        
        
        self.my_image_path = os.path.join('../images',partition+'2017')
        
        
        self.my_label_path=os.path.join('../labels/coco/labels',partition+'2017')
        
        self.transform = transform
        

    def __len__(self):
        return self.pointers.shape[0]

    def __getitem__(self, idx):
        img_path=os.path.join(self.my_image_path,self.pointers.iloc[idx, 0])
        label_path=os.path.join(self.my_label_path,self.pointers.iloc[idx, 1])
        try:
            with open(label_path) as box:
                box=box.read()
                box=pd.DataFrame([x.split() for x in box.rstrip('\n').split('\n')],columns=['class','xc','yc','w','h'])
        except FileNotFoundError:
            return None
        
        try:
            img = cv2.imread(img_path,1)
        except FileNotFoundError:
            print(self.pointers.iloc[idx, 0])
        
        b= box.values.astype(np.float32)
        b=torch.tensor(b)
        labels = b.T[0].reshape(b.shape[0], 1)
        one_hot_target = (labels == torch.arange(80).reshape(1, 80)).float()
        conf=torch.ones([b.shape[0],1])
        boxes=torch.cat((b.T[1:],conf.T,one_hot_target.T)).T
        
        sample={'images': img,
                'boxes': boxes,
               'img_name': self.pointers.iloc[idx, 1]}
        
        if self.transform:
            sample['images'] = self.transform(sample['images'])
            
        return sample
    
class ResizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        
        img = cv2.resize(img, (self.scale,self.scale))          #Resize to the input dimension
        img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H x W x C -> C x H x W 
        img_ = img_/255.0       #Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()     #Convert to float
        img_ = Variable(img_,requires_grad=False)# Convert to Variable
        
#         b_= box.values.astype(np.float32)
#         b_=torch.tensor(b)
#         labels_ = b_.T[0].reshape(b_.shape[0], 1)
#         one_hot_target_ = (labels_ == torch.arange(80).reshape(1, 80)).float()
#         conf_=torch.ones([b_.shape[0],1])
#         boxes_=torch.cat((b_.T[1:],conf_.T,one_hot_target_.T)).T
        return img_

    
class RandomHorizontalFlip(object):
    
    """Randomly horizontally flips the Image with the probability *p*"""

    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        if torch.rand([1]).item()>self.p:
            img=sample['images']
            bboxes=sample['boxes']

            img=sample['images']
            bboxes=sample['boxes']
            img =  img[:,::-1,:]
            bboxes[:,0] = 1 - bboxes[:,0]
            sample['boxes']=bboxes
            sample['images']=img

        return sample
    

class RandomVerticalFlip(object):
    
    """Randomly horizontally flips the Image with the probability *p*"""

    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        
        if torch.rand([1]).item()>self.p:
            img=sample['images']
            bboxes=sample['boxes']
            img=sample['images']
            bboxes=sample['boxes']
            img =  img[::-1,:,:]
            bboxes[:,1] = 1 - bboxes[:,1]
            sample['boxes']=bboxes
            sample['images']=img
        
        
        return sample
    
class RandomScale(object):
    """Randomly scales an image    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    scale: float or tuple(float)
        if **float**, the image is scaled by a factor drawn 
        randomly from a range (1 - `scale` , 1 + `scale`). If **tuple**,
        the `scale` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Scaled image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `xc,yc,w,h` of the box
        
    """

    def __init__(self, scale = 0.2, diff = False,clip_thres=0.25):
        self.scale = scale
        self.clip_thres=clip_thres
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff
        
        
    def __call__(self,sample):
        #Chose a random digit to scale by 
        img=sample['images']
        bboxes=sample['boxes']
        img_shape = img.shape
        
        if diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

        bboxes[:,:4] *= torch.tensor([resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y])
        canvas = np.zeros(img_shape, dtype = np.uint8)

        y_lim = int(min(resize_scale_y,1)*img_shape[0])
        x_lim = int(min(resize_scale_x,1)*img_shape[1])


        canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]

        img = canvas
        bboxes = helper.clip_box(bboxes, self.clip_thres)
        
        sample['images']=img
        sample['boxes']=bboxes
        
        return sample
    
    
    
class RandomTranslate(object):
    """Randomly Translates the image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    translate: float or tuple(float)
        if **float**, the image is translated by a factor drawn 
        randomly from a range (1 - `translate` , 1 + `translate`). If **tuple**,
        `translate` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Translated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `xc,yc,w,h` of the box
        
    """

    def __init__(self, translate = 0.2, diff = False,clip_thres=0.25):
        self.translate = translate
        self.clip_thres=clip_thres
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1


        else:
            assert self.translate > 0 & self.translate < 1
            self.translate = (-self.translate, self.translate)
            
            
        self.diff = diff
        
    def __call__(self, sample):        
        #Chose a random digit to scale by
        
        img=sample['images']
        bboxes=sample['boxes']
        img_shape = img.shape

        #translate the image

        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        
        if not diff:
            translate_factor_y = translate_factor_x

        canvas = np.zeros(img_shape).astype(np.uint8)


        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])


        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]

        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas

        bboxes[:,:2] += torch.tensor([translate_factor_x, translate_factor_y])
        bboxes = helper.clip_box(bboxes,self.clip_thres)

        sample['images']=img
        sample['boxes']=bboxes
        
        return sample
    
    