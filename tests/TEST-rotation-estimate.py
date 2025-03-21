from data import DataManager, Dataset, dataAugmenter
from network import Model

import matplotlib.pyplot as plt
import numpy as np

import torch as tt


if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    
    ''' Will save data as numpy arrays '''
    processor  = DataManager()

    ''' For Datasplitting '''
    here = processor.getParentDir()
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

    batch = next(iter(data_loader['train']))
    batch_images = batch['image']
    batch_rotations = batch['rotation']
    print(batch_rotations)
    estimated_rotations = Model().forward(batch_images)
    print(estimated_rotations)
    batch_images_derotated = dataAugmenter().rotateBatch(batch_images,-1E3*estimated_rotations)

    fig, axes = plt.subplots(2, 2, figsize=(6, 3))
    axes[0,0].imshow(batch_images[0,0,:,:], cmap="gray")
    axes[0,0].set_title(f"Batch Image 1")
    axes[0,0].axis("off")

    axes[1,0].imshow(batch_images[1,0,:,:], cmap="gray")
    axes[1,0].set_title(f"Batch Image 2")
    axes[1,0].axis("off")

    axes[0,1].imshow(batch_images_derotated[0,0,:,:], cmap="gray")
    axes[0,1].set_title(f"Batch Image 1 Rotated")
    axes[0,1].axis("off")

    axes[1,1].imshow(batch_images_derotated[1,0,:,:], cmap="gray")
    axes[1,1].set_title(f"Batch Image 2 Rotated")
    axes[1,1].axis("off")
    plt.show()
