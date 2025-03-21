from data import DataAugmenter, Dataset, DataManager
import numpy as np
import torch as tt

manager = DataManager()

here = manager.getParentDir()
path_logs = here/"output"/"logs"
path_weights = here/"output"/"weights"
path_plots = here/"output"/"plots"

images = np.load(here/"data"/"arrays"/"images.npy")
labels = np.load(here/"data"/"arrays"/"labels.npy")
splits = np.load(here/"data"/"splits"/"5-fold-indices.npz")

split_train = f"train_{'00'}"
split_val = f"val_{'00'}"

image = images[0][None, :]

shift_fraction = DataAugmenter().randomTuple((.25/2,.25/2))
image_resolution = DataAugmenter().getImageResolution(image)
shift_pixels = DataAugmenter().getImageShiftPixels(image_resolution,shift_fraction)
