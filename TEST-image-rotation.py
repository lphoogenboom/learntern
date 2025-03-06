from data import DataManager
from data import dataAugmenter
from data import Dataset
import numpy as np
import torch as tt
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tvf

if __name__ == "__main__":
	print("==== RAN AS FILE ====")

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

	data_set_train = Dataset(images[splits[split_train]],labels[splits[split_train]])
	data_set_val = Dataset(images[splits[split_val]],labels[splits[split_val]])

	# Define data loader
	data_loader_train = tt.utils.data.DataLoader(data_set_train, batch_size=5, shuffle=True) #Example
	data_loader_val = tt.utils.data.DataLoader(data_set_val, batch_size=5, shuffle=True) #Example
	data_loader = dict(train=data_loader_train,val=data_loader_val)

	test_images = next(iter(data_loader["train"]))
	test_image = test_images["image"][0]
	print(np.shape(test_image))

	augmenter = dataAugmenter()
	test_image_rotated = augmenter.rotateImage(test_image, 45)

	fig, axes = plt.subplots(1, 2, figsize=(6, 3))
	axes[0].imshow(test_image[0,:,:], cmap="gray")
	axes[0].set_title(f"Test Image")
	axes[0].axis("off")

	axes[1].imshow(test_image_rotated[0,:,:], cmap="gray")
	axes[1].set_title(f"Rotated Test Image")
	axes[1].axis("off")
	plt.show()