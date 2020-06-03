from __future__ import print_function, division
from torch.autograd import Variable
import numpy as np
import cv2
import os
import torch
import io
import pandas as pd
import helper as helper
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import imgaug as ia
from imgaug import parameters as iap
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

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
            img = cv2.imread(img_path,1)[:,:,::-1] 
            img_size=img.shape
        except FileNotFoundError:
            print(self.pointers.iloc[idx, 0])
        
        b= box.values.astype(np.float32)
#         b=torch.tensor(b)
#         labels = b.T[0].reshape(b.shape[0], 1)
#         one_hot_target = (labels == torch.arange(80).reshape(1, 80)).float()
#         conf=torch.ones([b.shape[0],1])
#         boxes=torch.cat((b.T[1:],conf.T,one_hot_target.T)).T
        
        sample={'images': img,
                'boxes': b,
               'img_name': self.pointers.iloc[idx, 1],
               'img_size':img_size}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ResizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        
        img=sample['images']
        b=sample['boxes']
        
        img = cv2.resize(img, (self.scale,self.scale))          #Resize to the input dimension
        img_ =  img.transpose((2,0,1))# H x W x C -> C x H x W 
        img_ = img_/255.0       #Add a channel at 0 (for batch) | Normalise
        img_ = torch.from_numpy(img_).float()     #Convert to float
        img_ = Variable(img_,requires_grad=False)# Convert to Variable
        
        b=torch.tensor(b).type(torch.FloatTensor)
        labels = b.T[0].reshape(b.shape[0], 1)
        one_hot_target = (labels == torch.arange(80).reshape(1, 80)).float()
        conf=torch.ones([b.shape[0],1])
        boxes=torch.cat((b.T[1:],conf.T,one_hot_target.T)).T
        
        sample['boxes']=boxes
        sample['images']=img_
        
        return sample

class Augment(object):
    
    def __init__(self):
        self.seq = iaa.Sequential([
        iaa.Sometimes(
            0.125,
            iaa.LinearContrast(alpha=(0.1, 1.9)),
            iaa.Fliplr(0.5)
        ),
        iaa.Sometimes(
            0.125,
            iaa.Grayscale(alpha=(0.1, 0.9)),
            iaa.Affine(
            translate_px={"y": (-150, 150)}
        )
        ),
        iaa.Sometimes(
            0.125,
            iaa.Solarize(0.5, threshold=(0, 256)),
            iaa.ShearX((-10, 10))
        ),
        iaa.Sometimes(
            0.125,
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.ShearY((-10, 10))
        ),
        iaa.Sometimes(
            0.125,
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            iaa.Affine(
            rotate=(-30, 30)
        )
        ),
        iaa.Sometimes(
            0.125,
            iaa.HistogramEqualization(),
            iaa.Affine(
            translate_px={"x": (-150, 150)}
        )
        ),
        iaa.Sometimes(
            0.125,
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
        )
        )
        
    ], random_order=True)
        
    def __call__(self, sample):
        
        bboxes=np.array([])
        while(bboxes.size==0):
            img=sample['images']
            bboxes=sample['boxes']

            bboxes=helper.convert2_abs(bboxes,img.shape)

            bbs = BoundingBoxesOnImage([
            BoundingBox(x1=b[1], y1=b[2], x2=b[3], y2=b[4], label=b[0]) for b in bboxes], shape=img.shape)

            image_aug, bbs_aug = self.seq(image=img, bounding_boxes=bbs)

            bbs_aug=bbs_aug.remove_out_of_image().clip_out_of_image()

            new_bboxes=bbs_aug.to_xyxy_array()

            labels=np.array([[box.label for box in bbs_aug.bounding_boxes]]).T
            new_bboxes=np.hstack((labels,new_bboxes))

            bboxes=helper.convert2_rel(new_bboxes,image_aug.shape)

        sample["boxes"]=bboxes
        sample['images']=image_aug
        
        return sample
