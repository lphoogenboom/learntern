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

learner = Learner(data_loader)
batch = next(iter(learner.dataloader['val']))
batch_images = batch['image']


straight_image = tt.tensor(images[0]).unsqueeze(0)
print(straight_image.shape)

batch_rotations = batch['rotation']
image_transformed =DataAugmenter().augmentImageAffine(
	straight_image,
	shift=(-5,5),
	angle=-90,
	scale=0.5
	)

image_corrected =DataAugmenter().augmentImageAffineInverse(
	image_transformed,
	shift=(-5,5),
	angle=-90,
	scale=0.5
	)

fig, axes = plt.subplots(3, 1, figsize=(6, 3))
axes[0].imshow(straight_image[0], cmap="gray")
axes[0].set_title(f"Batch Image")
axes[0].axis("off")

axes[1].imshow(image_transformed[0,:,:], cmap="gray")
axes[1].set_title(f"Tranformed Image")
axes[1].axis("off")

axes[2].imshow(image_corrected[0,:,:], cmap="gray")
axes[2].set_title(f"Corrected Image")
axes[2].axis("off")
fig.savefig("output/plots/rotations.png", dpi=300)#, bbox_inches="tight")