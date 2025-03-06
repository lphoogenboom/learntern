from pathlib import Path
from data import dataProcessor
import numpy as np
import torch as tt
import sys

# For datasplit

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
		return dict(image=self.images[index], label=self.labels[index])

if __name__ == "__main__":
	print("==== RAN AS FILE ====")
	processor = dataProcessor()
	pwd = processor.getParentDir()
	images = np.load(pwd/"data"/"arrays"/"images.npy")
	labels = np.load(pwd/"data"/"arrays"/"labels.npy")
	splits = np.load(pwd/"data"/"splits"/"5-fold-indices.npz")


	type = "val_00"
	dataset = Dataset(images[splits[type]],labels[splits[type]])
	print("break")
