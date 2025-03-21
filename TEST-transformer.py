import torch as tt
import numpy as np
import matplotlib.pyplot as plt
import random

from data import Dataset, DataManager, DataAugmenter

import torch as tt
import numpy as np
import matplotlib.pyplot as plt

from learner import Learner
from data import DataAugmenter,DataManager,Dataset

manager  = DataManager()

''' For Datasplitting '''
here = manager.getParentDir()
path_data = here/"data"/"arrays" # Original Data
path_splits = here/"data"/"splits" # Destination for splits

## Load Data Arrays
images = np.load(path_data/"images.npy") # Should be loaded already
labels = np.load(path_data/"labels.npy")
splits = np.load(here/"data"/"splits"/"5-fold-indices.npz")

split_train = f"train_00"
split_val = f"val_00"

data_set_train = Dataset(images[splits[split_train]],labels[splits[split_train]])
data_set_val = Dataset(images[splits[split_val]],labels[splits[split_val]])

data_loader_train = tt.utils.data.DataLoader(data_set_train, batch_size=2, shuffle=True) #Example
data_loader_val = tt.utils.data.DataLoader(data_set_val, batch_size=2, shuffle=True) #Example
data_loader = dict(train=data_loader_train,val=data_loader_val)

idx =random.randint(0,100)

image = tt.from_numpy(images[idx][None,:,:])

augmenter = DataAugmenter()

x,y = augmenter.getImageResolution(image)

# Get random translation
shift_fraction = augmenter.randomTuple((.25/2,.25/2))
image_resolution = augmenter.getImageResolution(image)
pixels_x, pixels_y = augmenter.getImageShiftPixels(image_resolution,shift_fraction)
shift_pixels = np.array([pixels_x, pixels_y])
# Get random angle
angle = random.uniform(-np.pi,np.pi)
# Get random scale
scale  = random.uniform(0.8,1.2)

# Get Transforms
translation_transform = augmenter.getTranslationTransform(shift_pixels)
rotation_transform = augmenter.getCenterRotationTransform(angle)
scale_transform = augmenter.getCenterScaleTransform(scale)

transform = augmenter.combineTransorms(rotation_transform,scale_transform,translation_transform)
image_augmented = augmenter.applyTransform(image,transform)


translation_transform = augmenter.getTranslationTransform(-shift_pixels)
rotation_transform = augmenter.getCenterRotationTransform(-angle)
scale_transform = augmenter.getCenterScaleTransform(1/scale)

transform = augmenter.combineTransorms(translation_transform,scale_transform,rotation_transform)
image_corrected = augmenter.applyTransform(image_augmented,transform)

fig, axes = plt.subplots(3, 1, figsize=(6, 3))
axes[0].imshow(image[0], cmap="gray")
axes[0].set_title(f"Batch Image")
axes[0].axis("off")

axes[1].imshow(image_augmented[0], cmap="gray")
axes[1].set_title(f"Augmented Image")
axes[1].axis("off")

axes[2].imshow(image_corrected[0], cmap="gray")
axes[2].set_title(f"Augmented Image")
axes[2].axis("off")
fig.savefig("output/plots/transformer-test.png", dpi=300, bbox_inches="tight")

