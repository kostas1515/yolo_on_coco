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

    def __init__(self, partition,coco_version, transform=None):
        """
        Args:
            zip_file (string): Path to the zip file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.pointers=pd.read_csv('../pointers/'+partition+coco_version+'.txt',names=['img'])
        self.pointers['box']=self.pointers['img'].apply(lambda x: x.split('.')[0]+'.txt')
        if coco_version=='2017':
            self.my_image_path = os.path.join('../images',partition+'2017/')
            self.my_label_path=os.path.join('../labels/coco/labels',partition+'2017/')
        elif coco_version=='2014':
            self.my_image_path = '../images'
            self.my_label_path='../labels/coco/labels'
        
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
        
        sample={'images': [img],
                'boxes': [b],
               'img_name': self.pointers.iloc[idx, 1].split('/')[-1],
               'img_size':img_size}
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ResizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        
        imgs=sample['images']
        bbs=sample['boxes']
        
        imgs = [cv2.resize(i, (self.scale,self.scale)) for i in imgs]         #Resize to the input dimension
        imgs =  [i.transpose((2,0,1)) for i in imgs]# H x W x C -> C x H x W
        
        imgs = [i/255.0 for i in imgs]       #Add a channel at 0 (for batch) | Normalise.

        imgs = [torch.from_numpy(i).float() for i in imgs]     #Convert to float
        mean=torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std=torch.tensor([[[0.229, 0.224, 0.225]]]).T
        imgs = [(i-mean)/std for i in imgs]
        
        bbs=[torch.tensor(b).type(torch.FloatTensor) for b in bbs]
        labels = [b.T[0].reshape(b.shape[0], 1) for b in bbs]
        one_hot_target = [(l == torch.arange(80).reshape(1, 80)).float() for l in labels]
        conf=[torch.ones([b.shape[0],1]) for b in bbs]
        bbs=[torch.cat((b.T[1:],c.T,oh.T)).T for b,c,oh in zip(bbs,conf,one_hot_target)]
        
        sample['boxes']=bbs
        sample['images']=imgs
        
        return sample

class Augment(object):
    
    def __init__(self,num_of_augms=1):
        self.num_of_augms=num_of_augms
        self.aug=iaa.OneOf([
            iaa.Sequential([
                iaa.LinearContrast(alpha=(0.1, 1.9)),
                iaa.Fliplr(0.5)
            ]),
            iaa.Sequential([
                iaa.Grayscale(alpha=(0.1, 0.9)),
                iaa.Affine(
                translate_px={"y": (-150, 150)}
            )
            ]),
            iaa.Sequential([
                iaa.Solarize(0.5, threshold=(0, 256)),
                iaa.ShearX((-10, 10))
            ]),
            iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 0.5)),
                iaa.ShearY((-10, 10))
            ]),
            iaa.Sequential([
                iaa.Multiply((0.8, 1.2), per_channel=0.2),
                iaa.Affine(
                rotate=(-30, 30)
            )
            ]),
            iaa.Sequential([
                iaa.HistogramEqualization(),
                iaa.Affine(
                translate_px={"x": (-150, 150)}
            )
            ]),
            iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 0.5)),
                iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
            )
            ])
        ])
        
    def __call__(self, sample):
        
        bbox_list=sample["boxes"]
        img_list=sample['images']

        temp_img_=img_list[0]
        temp_b_=bbox_list[0]
        
        for i in range(self.num_of_augms):
            bboxes=np.array([])
            
            while(bboxes.size==0):
                bboxes=helper.convert2_abs(temp_b_,temp_img_.shape)

                bbs = BoundingBoxesOnImage([
                BoundingBox(x1=b[1], y1=b[2], x2=b[3], y2=b[4], label=b[0]) for b in bboxes], shape=temp_img_.shape)

                image_aug, bbs_aug = self.aug(image=temp_img_, bounding_boxes=bbs)

                bbs_aug=bbs_aug.remove_out_of_image().clip_out_of_image()

                new_bboxes=bbs_aug.to_xyxy_array()

                labels=np.array([[box.label for box in bbs_aug.bounding_boxes]]).T
                new_bboxes=np.hstack((labels,new_bboxes))

                bboxes=helper.convert2_rel(new_bboxes,image_aug.shape)
                temp_b_=helper.convert2_rel(temp_b_,temp_img_.shape)
                
            bbox_list.append(bboxes)
            img_list.append(image_aug)
            
        sample["boxes"]=bbox_list
        sample['images']=img_list
        
        return sample
