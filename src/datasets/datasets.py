import cv2
import torch
import pydicom 

import numpy as np
import albumentations as albu

from PIL import Image


from preprocess.preprocess import normalize

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

class MaskGenDataset(torch.utils.data.Dataset):
    def __init__(self, meta_data, 
                 transform=None,
                 preprocessing=None,
                 classes=None, 
                 augmentation=None, 
                ):
        self.meta_data = meta_data.reset_index()
        self.transforms = transform
        self.classes = classes
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.imgsize = 512
        self.resize =  albu.Compose([
                                  albu.Resize(height = self.imgsize, width = self.imgsize),
                               ])
        self.to_tensor = albu.Compose([
                                  albu.Lambda(image=  to_tensor, mask=to_tensor),
                               ])
        
        
    def __getitem__(self, index):
        # 읽어오기 -> numpy변환 -> 가운데만 선택
        margin = 50
        image = np.array(pydicom.dcmread(\
                         self.meta_data.loc[index, '1.원본'])\
                         .pixel_array)\
                         [margin:-margin,margin:-margin]
        ###
        image = normalize(image)
        image = image.astype(np.float32)[..., np.newaxis]

        # 읽어오기 -> numpy변환 -> 가운데만 선택 -> 합치기
        musle = np.array(Image.open(self.meta_data.loc[index, '2.근육']))[margin:-margin,margin:-margin, np.newaxis]
        visceral = np.array(Image.open(self.meta_data.loc[index, '3.내장지방']))[margin:-margin,margin:-margin, np.newaxis]
        subcutaneous = np.array(Image.open(self.meta_data.loc[index, '4.피하지방']))[margin:-margin,margin:-margin, np.newaxis]
        target_mask = np.concatenate([musle, visceral, subcutaneous], axis = 2)
        
        #append backgraound chennel
        background = target_mask.sum(axis = 2) #(128, 128, 채널(musle, fat, innerfat))
        background = np.where(background != 0, 0, 1)[...,np.newaxis]
        target_mask = np.concatenate([target_mask, background], axis = 2)
        
        # Preprocessing----------------------------------------
        # Resize -> augmentation -> preprocessing(pretrained 모델) -> to tensor
        # resize (모두 적용)
        sample = self.resize(image=image, mask=target_mask)
        image, target_mask = sample['image'], sample['mask']

        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=target_mask)
            image, target_mask = sample['image'], sample['mask']

        # reshape for converting to tensor (모두 적용)
        sample = self.to_tensor(image=image, mask=target_mask)
        image, target_mask = sample['image'], sample['mask']
        return image, target_mask
        #return torch.tensor(image, device = device), torch.tensor(target_mask, device = device)
    
    def __len__(self):
        return len(self.meta_data)
    
class CT_Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path=None,
                 imglist = None,
                 preprocessing=None,
                 classes=None, 
                 augmentation=None, 
                ):
        self.path = path
        self.imglist = imglist
        self.classes = classes
        self.preprocessing = preprocessing
        self.augmentation = augmentation
        self.to_tensor = albu.Compose([
                                  albu.Lambda(image=  to_tensor, mask=to_tensor),
                               ])
    def __len__(self):
        return len(self.imglist)
        
        
    def __getitem__(self,index):
        # 읽어오기 -> numpy변환 -> 가운데만 선택
        
        imgname = self.imglist[index]
        
        img = cv2.imread(f"{self.path}/img/{imgname}",
                         cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(f"{self.path}/mask/{imgname[:-7]+'mask.png'}",
                          cv2.IMREAD_GRAYSCALE)

        img = (img / 255)[..., np.newaxis]

        target = np.zeros([mask.shape[0],
                           mask.shape[1],
                           4])

        target[...,0] = mask == 0
        target[...,1] = mask == 1
        target[...,2] = mask == 2
        target[...,3] = mask == 3


        #apply augmentation
        if self.augmentation:
            sample = self.augmentation(image=img, mask=target)
            img, target = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=target)
            img, target = sample['image'], sample['mask']

        # reshape for converting to tensor (모두 적용)
        sample = self.to_tensor(image=img, mask=target)
        img, target = sample['image'], sample['mask']
        return img, target
    