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
               'img_name': self.pointers.iloc[idx, 1]}
        
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
        img_ =  img.transpose((2,0,1))  # BGR -> RGB | H x W x C -> C x H x W 
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
            assert (self.translate > 0.0) & (self.translate < 1.0)
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
    
class RandomRotate(object):
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, angle = 10,clip_thres=0.25):
        self.angle = angle
        self.clip_thres=clip_thres
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, sample):
        
        img=sample['images']
        bboxes=sample['boxes']


        #Chose a random digit to scale by 
        angle = random.uniform(*self.angle)

        w,h = img.shape[1], img.shape[0]
        cx, cy = w/2, h/2
        img = helper.rotate_im(img, angle)
        
        #transform relative to absolute and xmin xma ymin ymax
        bboxes[:,0]=bboxes[:,0]*w
        bboxes[:,1]=bboxes[:,1]*h
        bboxes[:,2]=bboxes[:,2]*w
        bboxes[:,3]=bboxes[:,3]*h
        bboxes[:,0]=bboxes[:,0]-bboxes[:,2]/2
        bboxes[:,1]=bboxes[:,1]-bboxes[:,3]/2
        bboxes[:,2]=bboxes[:,0]+bboxes[:,2]
        bboxes[:,3]=bboxes[:,1]+bboxes[:,3]
        
        
        corners = helper.get_corners(bboxes)
        corners = torch.cat([corners, bboxes[:,4:]],dim=1)

        corners[:,:8] = helper.rotate_box(corners[:,:8], angle, cx, cy, h, w)

        new_bbox = helper.get_enclosing_box(corners)


        scale_factor_x = img.shape[1] / w

        scale_factor_y = img.shape[0] / h

        img = cv2.resize(img, (w,h))

        new_bbox[:,0:4] = new_bbox[:,0:4] / torch.tensor([scale_factor_x, scale_factor_y,scale_factor_x, scale_factor_y])


        bboxes  = new_bbox

        bboxes = helper.clip_box(bboxes, self.clip_thres)

        sample['images']=img
        sample['boxes']=bboxes
        
        return sample
    
    
    
class RandomShear(object):
    """Randomly shears an image in horizontal direction   
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    shear_factor: float or tuple(float)
        if **float**, the image is sheared horizontally by a factor drawn 
        randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
        the `shear_factor` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, shear_factor = 0.2):
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
        
        
    def __call__(self, sample):
        
        img=sample['images']
        bboxes=sample['boxes']

        shear_factor = random.uniform(*self.shear_factor)

        w,h = img.shape[1], img.shape[0]

        if shear_factor < 0:
            sample = RandomHorizontalFlip(-1)(sample)
            img=sample['images']
            bboxes=sample['boxes']

        M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

        nW =  img.shape[1] + abs(shear_factor*img.shape[0])

        bboxes[:,[0,2]] += ((bboxes[:,[1,3]]) * abs(shear_factor) ).astype(int) 


        img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

        if shear_factor < 0:
            sample = RandomHorizontalFlip(-1)(sample)
            img=sample['images']
            bboxes=sample['boxes']

        img = cv2.resize(img, (w,h))

        scale_factor_x = nW / w

        bboxes[:,:4] /= [scale_factor_x, 1, scale_factor_x, 1] 


        return img, bboxes
