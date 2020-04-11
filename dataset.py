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
from zipfile import ZipFile

class Coco(Dataset):

    def __init__(self, partition, transform=None):
        """
        Args:
            zip_file (string): Path to the zip file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.pointers=pd.read_csv('pointers/'+partition+'2017.txt',names=['img'])
        self.pointers['box']=self.pointers['img'].apply(lambda x: 'coco/labels/'+x.split('.')[0]+'.txt')
        
        
        self.my_image_zip = ZipFile(os.path.join('images',partition+'2017.zip'), 'r')
        self.img_handle = None
        
        
        self.my_label_zip=ZipFile(os.path.join('labels','coco2017labels.zip'), 'r')
        self.label_handle = None
        
        self.transform = transform
        

    def __len__(self):
        return self.pointers.shape[0]

    def __getitem__(self, idx):

        if self.label_handle is None:
            try:
                with self.my_label_zip.open(self.pointers.iloc[idx, 1]) as box:
                    box=box.read().decode("utf-8")
                    box=pd.DataFrame([x.split() for x in box.rstrip('\n').split('\n')],columns=['class','xc','yc','w','h'])
            except KeyError:
                return None
        
        if self.img_handle is None:
            try:
                with self.my_image_zip.open(self.pointers.iloc[idx, 0]) as buffer:
                    buffer=buffer.read()
                    img = cv2.imdecode(np.frombuffer(buffer, np.uint8),1)
            except KeyError:
                print(self.pointers.iloc[idx, 0])
        
        b= box.values.astype(np.float32)
        b=torch.tensor(b)
        labels = b.T[0].reshape(b.shape[0], 1)
        one_hot_target = (labels == torch.arange(80).reshape(1, 80)).float()
        conf=torch.ones([b.shape[0],1])
        boxes=torch.cat((b.T[1:],conf.T,one_hot_target.T)).T
        
        if self.transform:
            img = self.transform(img)
        return {'images': img,
                'boxes': boxes}
    
class ResizeToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, img):
        
        img = cv2.resize(img, (self.scale,self.scale))          #Resize to the input dimension
        img_ =  img[:,:,::-1].transpose((2,0,1))  # BGR -> RGB | H X W C -> C X H X W 
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

    
