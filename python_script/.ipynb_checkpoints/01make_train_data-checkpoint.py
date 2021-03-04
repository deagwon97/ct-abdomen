import os
import sys
sys.path.append("../src/")
import re
import torch
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensor
from PIL import Image
from tqdm import tqdm
import cv2
from utils.utils import create_meta_df
from datasets.datasets import MaskGenDataset
import multiprocessing
    

if __name__ == "__main__":
    
    # input path
    #if sys.argv[1] == '-raw_data':
    #    raw_data = sys.argv[2]
    raw_data = "../raw_data/"


    images_paths = create_meta_df(raw_data)
    images_paths = images_paths.dropna().reset_index(drop=True)

    CLASSES = 4
    dataset = MaskGenDataset(
                images_paths,
                augmentation = None, 
                preprocessing = None,
                classes=CLASSES,
            )

    test_dataset = MaskGenDataset(
                images_paths[:len(images_paths) // 10],
                augmentation = None, 
                preprocessing = None,
                classes=CLASSES,
            )



    
    cpu_num = multiprocessing.cpu_count()
    step = 1000
    
    
    # preprocess train data
    idxs = range(dataset.__len__() - 1)
    path = "../preprocess_data/train_preprocess"
    def train_f(idx):
        sample = dataset[idx]
        img = (sample[0] * 255).astype(np.uint8)[0]
        mask = sample[1]
        mask = np.argmax(mask, axis = 0).astype(np.uint8)
        cv2.imwrite(f"{path}/img/{str(idx).zfill(5)}_img.png", img)
        cv2.imwrite(f"{path}/mask/{str(idx).zfill(5)}_mask.png", mask)
        
    

    for j in tqdm(range(0, len(idxs), step)):
        pool = multiprocessing.Pool(processes=cpu_num)
        pool.map(train_f, idxs[j:j+step])
        pool.close()
        pool.join()


    # preprocess test data
    idxs = range(test_dataset.__len__() - 1)
    path = "../preprocess_data/test_preprocess"
    def test_f(idx):
        sample = test_dataset[idx]
        img = (sample[0] * 255).astype(np.uint8)[0]
        mask = sample[1]
        mask = np.argmax(mask, axis = 0).astype(np.uint8)
        cv2.imwrite(f"{path}/img/{str(idx).zfill(5)}_img.png", img)
        cv2.imwrite(f"{path}/mask/{str(idx).zfill(5)}_mask.png", mask)

    for j in tqdm(range(0, len(idxs), step)):
        pool = multiprocessing.Pool(processes=cpu_num)
        pool.map(test_f, idxs[j:j+step])
        pool.close()
        pool.join()