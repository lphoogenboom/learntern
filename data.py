from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf
import torch as tt
import random
import json

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

class DataAugmenter():
     
    def __init__(self):
        self.configuration = json.load(open("./configuration.json", "r"))
        self.device = self.configuration['device']
        return
    
    def getTranslationTransform(self, translation):
        transform = tt.cat([
            tt.eye(2),
            tt.tensor([[-translation[0]],[-translation[1]]])],
            dim=1
            )
        return transform
    
    def getCenterRotationTransform(self,angle, image=np.zeros((28,28))):
        cos = np.cos(angle)
        sin = np.sin(angle)
        W, H = self.getImageResolution(image)
        c_x = (W-0.5)/2
        c_y = (H-0.5)/2

        transform = tt.tensor(
            [[cos,-sin,-c_x*(cos)+c_y*sin+c_x],
            [sin,cos,-c_y*(cos)-c_x*sin+c_y]],
            dtype=tt.float32
        )
        return transform
    
    def getCenterScaleTransform(self, scale, image=np.zeros((28,28))):
        W,H = self.getImageResolution(image)
        c_x = (W-0.5)/2
        c_y = (H-0.5)/2
        transform  = tt.tensor([
            [scale,0,(1-scale)*c_x],
            [0,scale,(1-scale)*c_y]
            ],
            dtype=tt.float32)
        return transform
    
    def combineTransorms(self, *transforms):

        combined_transform = tt.eye(3,dtype=tt.float32)
        row_3 = tt.tensor([[0,0,1]],dtype=tt.float32)

        for transform in transforms:
            transform = tt.cat([
                transform,
                row_3],
                dim=0
            )
            combined_transform @= transform
        
        combined_transform = combined_transform[:-1,:]
        return combined_transform
    
    def invertTransform(self, transform):
        row_3 = tt.tensor([[0,0,1]], dtype=tt.float32)
        transform_3d = tt.cat([transform,row_3],dim=0)
        inverse = tt.linalg.inv(transform_3d)
        return inverse[:-1,:]
    
    def getTargetTensor(self, W:int, H:int):
        x = tt.arange(W)
        y = tt.arange(H)
        X, Y = tt.meshgrid(y, x, indexing='ij')

        X_flat = X.flatten().float()
        Y_flat = Y.flatten().float()
        ones = tt.ones_like(X_flat)
        target = tt.stack([X_flat,Y_flat,ones], dim=0)
        return target

    
    def bilinearInterpolate(self,image,x,y):
        W, H = self.getImageResolution(image)

        # Check if sample is on image plane
        if x< 0 or x>(W-1) or y<0 or y>(H-1):
            return 0.0
        
        # floor and ceil of x
        x0 = int(tt.floor(x))
        x1 = x0 + 1
        # floor and ceil of y
        y0 = int(tt.floor(y))
        y1 = y0 + 1

        # Clamp so we don't go out of bounds
        if x1 >= W: 
            x1 = W - 1
        if y1 >= H: 
            y1 = H - 1

        # Get the fractional part (how far between x0 and x1?)
        dx = x - x0
        dy = y - y0

        # Pixel values at corners
        v00 = image[0,x0, y0]
        v01 = image[0,x0, y1]
        v10 = image[0,x1, y0]
        v11 = image[0,x1, y1]

        # bilinear interpolation formula
        top = v00 * (1 - dx) + v01 * dx      # row y0
        bot = v10 * (1 - dx) + v11 * dx      # row y1
        pixel_val = top * (1 - dy) + bot * dy
        return pixel_val
    
    def applyTransform(self,image,transform):

        W, H = self.getImageResolution(image)
        target = self.getTargetTensor(W,H)

        image_transformed = tt.zeros((2,W*H), dtype=tt.float32)

        source = transform@target
        image_transformed = self.bilinearInterpolateVector(image, source)
        return image_transformed
    
    def bilinearInterpolateVector(self, image, source_coordinates):

        sampled_output = tt.zeros([1,28,28], dtype=tt.float32)
        W, H = self.getImageResolution(image)

        inside_mask = (source_coordinates[1,:] >= 0) & (source_coordinates[1,:] <= W - 1) & (source_coordinates[0,:] >= 0) & (source_coordinates[0,:] <= H - 1)

        # Clamp for if source coodinates are in fractional terms
        source_x_clamped = tt.clamp(source_coordinates[1,:],0 , W-1)
        source_y_clamped = tt.clamp(source_coordinates[0,:],0 , H-1)

        # Grid corner coordinates for source sampling
        x_left = tt.floor(source_x_clamped).long()
        y_top = tt.floor(source_y_clamped).long()
        x_right = tt.clamp(x_left+1, 0, W-1)
        y_bottom = tt.clamp(y_top+1, 0, H-1)

        # left-distance of gridpoints for interpolation
        dx = source_x_clamped - x_left
        dy = source_y_clamped - y_top

        # Sample source image at grid corners
        image_top_left = image[0,y_top,x_left]
        image_top_right = image[0,y_top,x_right]
        image_bottom_left = image[0,y_bottom,x_left]
        image_bottom_right = image[0,y_bottom,x_right]

        # Interpolate samples to grid points
        wa = (1 - dx) * (1 - dy)
        wb = dx * (1 - dy)
        wc = (1 - dx) * dy
        wd = dx * dy
        test = wa * image_top_left + wb * image_top_right + wc * image_bottom_left + wd * image_bottom_right
        test[~inside_mask] = 0.0
        test = test.reshape(H,W)
        return test.unsqueeze(0)

    def getImageResolution(self, image):
        W = image.shape[-2]
        H = image.shape[-1]
        return W, H
    
    def getImageShiftPixels(self, image_resolution, shift_fraction):
        pixels_x = int(np.floor(shift_fraction[0]*image_resolution[0]))
        pixels_y = int(np.floor(shift_fraction[1]*image_resolution[1]))
        return pixels_x, pixels_y
    
    def randomTuple(self, bounds:tuple[float,float]=(1.,1.)):
        ### Random tuple on bounded square around (0,0)
        dx = random.uniform(-bounds[0],bounds[0])
        dy = random.uniform(-bounds[1],bounds[1])
        return dx, dy

class Dataset(tt.utils.data.Dataset):

    def __init__(self,images,labels):
        self.images = images  # (Channel, Height, Width)
        self.images = self.images[:, None]  # (Batch, 1, Height, Width) required by torch
        self.images = tt.from_numpy(self.images)  # Convert to tensor

        # Standardise images
        self.images = self.images.float()
  
        self.images -= images.mean()
        self.images /= images.std()

        self.labels = tt.from_numpy(labels)
        self.augmenter = DataAugmenter()
        self.theta = dict()
        # self.labels = tt.nn.functional.one_hot(self.labels.long(), num_classes=10)  # 1-hot encoding so neural network can have 10 binary outputs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index): # get image-label pair
        image = self.images[index]
        # print(image.device)

        # Get random translation
        shift_fraction = self.augmenter.randomTuple((.25/2,.25/2))
        image_resolution = self.augmenter.getImageResolution(image)
        pixels_x, pixels_y = self.augmenter.getImageShiftPixels(image_resolution,shift_fraction)
        shift_pixels = np.array([pixels_x, pixels_y])
        # Get random angle
        angle = random.uniform(-np.pi,np.pi)

        # Get random scale
        scale  = random.uniform(0.8,1.2)

        # Get Transforms
        translation_transform = self.augmenter.getTranslationTransform(shift_pixels)
        rotation_transform = self.augmenter.getCenterRotationTransform(angle)
        scale_transform = self.augmenter.getCenterScaleTransform(scale)

        transform = self.augmenter.combineTransorms(rotation_transform,scale_transform,translation_transform)

        image_augmented = self.augmenter.applyTransform(image,transform)

        return dict(image=image_augmented, label=self.labels[index],transform = transform)