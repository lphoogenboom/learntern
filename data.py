from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf
import torch as tt
import random

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import torchvision.transforms.functional as tvf
import torch.nn.functional as tf

import matplotlib.pyplot as plt

class DataManager():

    def getParentDir(self): # Gets PWD
        parent_dir = Path(__file__).resolve().parent
        return parent_dir

    def unzipData(self, zipPath): # unzip archived files
        if Path('data/csv/mnist_test.csv').is_file():
            print('At least 1 unzipped file exist... \nWill not unzip until target directory is empty.')
            return
        
        basePath = self.getParentDir()
        relativePath = zipPath.split("/") # path from pwd to .zip (including file)

        # append relative path to base path
        targetPath = basePath
        for dir in relativePath: 
            targetPath = targetPath/dir

        with zf.ZipFile(targetPath, 'r') as archive:
            archive.extractall('data/csv')

    def csvLoad(self, csvPath): # load csv and seperate images from labels
        targetPath = self.getParentDir()
        path = csvPath.split("/")
        for dir in path:
            targetPath = targetPath/dir
        data = pd.read_csv(targetPath)
        return data
    
    def trainTestSplit(self,labels): # Creates split as INDEX Dictionary
        datasplit = dict()
        idx_train, idx_test, _, _ = train_test_split(
            range(len(labels)),
            labels,
            test_size=0.2,
            stratify=labels,
        )
        datasplit["train"] = idx_train
        datasplit["test"] = idx_test
        return datasplit

    def kFoldSplit(self, data, indices, key, k): # k-fold split as INDICES
        skf_train_val = StratifiedKFold(n_splits=k, shuffle=True)
        for i, (idx_train, idx_va) in enumerate(skf_train_val.split(indices[key], data[indices[key]])):
            indices[f"train_{i:02d}"] = idx_train
            indices[f"val_{i:02d}"] = idx_va
        return indices

class dataAugmenter():
     
     def __init__(self):
        return
     
     def rotateImage(self, *args, **kwargs): # Expected image format: [channel, width, height]
        image_rotated = tvf.rotate(*args, **kwargs)
        return image_rotated
     
     def rotateBatch(self,batch,angles):
        angles = tt.tensor(angles)
        if angles.dtype != tt.float:
            angles = angles.float()

        angles=tt.deg2rad(angles)
        rotation_matrices = tt.stack([
            tt.tensor([
                [tt.cos(angle), -tt.sin(angle), 0],
                [tt.sin(angle), tt.cos(angle), 0]
            ]) for angle in angles
        ])
        
        # Create grid and apply rotation
        grid = tf.affine_grid(rotation_matrices, batch.size(), align_corners=True)
        rotated_batch = tf.grid_sample(batch, grid, align_corners=True)
        return rotated_batch
     

class Dataset(tt.utils.data.Dataset):

    def __init__(self,images,labels):
        self.images = images  # (Batch, Height, Width)
        self.images = self.images[:, None]  # (Batch, 1, Height, Width) required by torch
        self.images = tt.from_numpy(self.images)  # Convert to tensor

        # Normalise images
        self.images = self.images.float()
        self.images -= tt.min(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None] # subtract minimum
        self.images /= tt.max(self.images.flatten(start_dim=1), dim=1).values[:, None, None, None] # Divide by maximum

        self.labels = tt.from_numpy(labels)
        self.labels = tt.nn.functional.one_hot(self.labels.long(), num_classes=10)  # 1-hot encoding so neural network can have 10 binary outputs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index): # get image-label pair
    # augment image
        dxdy = 0
        angle = random.randint(0,359)
        rotated_image = dataAugmenter().rotateImage(self.images[index], angle)
        angle = tt.nn.functional.one_hot(tt.tensor(angle),num_classes=360)
        return dict(image=rotated_image, label=self.labels[index],rotation=angle ,translation=dxdy)