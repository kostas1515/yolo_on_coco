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

    def __init__(self, partition,coco_version,subset=1.0, transform=None):
        """
        Args:
            zip_file (string): Path to the zip file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.pointers=pd.read_csv('../pointers/'+partition+coco_version+'.txt',names=['img'])
#         self.pointers=pd.read_csv('../pointers/subset2017.txt',names=['img'])
        self.pointers['box']=self.pointers['img'].apply(lambda x: x.split('.')[0]+'.txt')
        if coco_version=='2017':
            self.my_image_path = os.path.join('../images',partition+'2017/')
            self.my_label_path=os.path.join('../labels/coco/labels',partition+'2017/')
        elif coco_version=='2014':
            self.my_image_path = '../images'
            self.my_label_path='../labels/coco/labels'
        if subset<1:
            self.pointers=self.pointers.sample(n=int(self.pointers.shape[0]*subset), random_state=int(np.random.randint(0,10,size=1)))
        self.transform = transform
        

    def __len__(self):
        return self.pointers.shape[0]

    def __getitem__(self, idx):
        img_path=os.path.join(self.my_image_path,self.pointers.iloc[idx, 0])
        label_path=os.path.join(self.my_label_path,self.pointers.iloc[idx, 1])
        
        with open(label_path) as box:
            box=box.read()
            box=pd.DataFrame([x.split() for x in box.rstrip('\n').split('\n')],columns=['class','xc','yc','w','h'])
        

        img = cv2.imread(img_path,1)[:,:,::-1] 
        img_size=img.shape

        
        b= box.values.astype(np.float32)
#         b=torch.tensor(b)
        
        b[:,1:]=helper.convert2_abs_xyxy(b[:,1:],img_size)
        
        
        sample={'image': img,
                'boxes': b[:,1:],
                'area': None,
                'labels':b[:,0],
                'img_size':img_size}
        
        if self.transform:
            sample = self.transform(sample)
        
        targets={'boxes': sample['boxes'],
            'labels':sample['labels'],
            'image_id': torch.tensor(int(self.pointers.iloc[idx, 1].split('/')[-1].split('.')[0]),dtype=torch.int64),
            'area' :sample['area'],
            'iscrowd':torch.zeros(sample['boxes'].shape[0],dtype=torch.uint8),
            'img_size':img_size}
            
        return sample['image'],targets
    
class ResizeToTensor(object):
    """
    Image: Resizes and normalizes it to predefined yolo dimensions. Also it transposes it and adds extra dimension for batch.
    BOXES: Transformes them from absolute X0Y0X1Y1 to yolo dimension relative XCYCWH -> range: [0, 1]
    Also it calculate bbox area and put ist in the sample
    """
    

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        
        img=sample['image']
        bbs=sample['boxes']
        labels=sample['labels']
        
        bbs=torch.tensor(bbs)
        labels=torch.tensor(labels,dtype=torch.int64)
        
        img = cv2.resize(img, (self.scale,self.scale))        #Resize to the input dimension
        img =  img.transpose((2,0,1)) # H x W x C -> C x H x W
        img = img/255.0       #Add a channel at 0 (for batch) | Normalise.
        
        img=torch.tensor(img,dtype=torch.float)
        mean=torch.tensor([[[0.485, 0.456, 0.406]]]).T
        std=torch.tensor([[[0.229, 0.224, 0.225]]]).T
        img = (img-mean)/std
        
        img=img.unsqueeze(0) #Add a channel at 0 (for batch)
        
        
        bbs=helper.convert2_rel_xcycwh(bbs,sample['img_size'])
        
        area=(bbs[:,2])*(bbs[:,3])
        sample["boxes"]=bbs
        sample["labels"]=labels
        sample['image']=img
        sample['area']=area
        
        return sample

class Augment(object):
    #use mosaic
    # TO DO FIX IT
    
    def __init__(self,num_of_augms=0):
        self.num_of_augms=num_of_augms
        self.aug=iaa.OneOf([
            iaa.Sequential([
                iaa.LinearContrast(alpha=(0.75, 1.5)),
                iaa.Fliplr(0.5)
            ]),
            iaa.Sequential([
                iaa.Grayscale(alpha=(0.1, 0.9)),
                iaa.Affine(
                translate_percent={"y": (-0.15, 0.15)}
            )
            ]),
            iaa.Sequential([
                iaa.Solarize(0.5, threshold=(0, 256)),
                iaa.ShearX((-10, 10))
            ]),
            iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 1)),
                iaa.ShearY((-10, 10))
            ]),
            iaa.Sequential([
                iaa.Multiply((0.5, 1.5), per_channel=0.25),
                iaa.Fliplr(0.5),
            ]),
            iaa.Sequential([
                iaa.HistogramEqualization(),
                iaa.Affine(
                translate_percent={"x": (-0.25, 0.25)},
                    shear=(-8, 8)
            )
            ]),
            iaa.Sequential([
                iaa.Crop(percent=(0.01, 0.1)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
            )
            ]),
            iaa.Sequential([
                iaa.GaussianBlur(sigma=(0, 0.9)),
                iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}
            )
            ])
        ])
                        
    def __call__(self, sample):
        
        temp_img_=sample["image"]
        temp_b_=sample["boxes"]
        labels=sample["labels"]
        
        at_least_one_box=False
        
        
        
        while(at_least_one_box==False):
            bboxes=temp_b_

            bbs = BoundingBoxesOnImage([
            BoundingBox(x1=b[0], y1=b[1], x2=b[2], y2=b[3], label=l) for b,l in zip(temp_b_,labels)], shape=temp_img_.shape)
            
            image_aug, bbs_aug = self.aug(image=temp_img_, bounding_boxes=bbs)

            bbs_aug=bbs_aug.remove_out_of_image().clip_out_of_image()
            
            
            new_bboxes=bbs_aug.to_xyxy_array()
            new_labels=np.array([box.label for box in bbs_aug.bounding_boxes])
            
            if(new_labels.size>0):
                at_least_one_box=True
            
        sample["boxes"]=new_bboxes
        sample['images']=image_aug
        sample['labels']=new_labels
        
        return sample
