from pathlib import Path
from dataProcessor import dataProcessor
import numpy as np
import torch as tt
import sys

# For datasplit


class Dataset(tt.utils.data.Dataset):

	def __init__(self,images,labels):
		self.images = images  # (Batch, Height, Width)
		self.images = self.images[:, None]  # (Batch, 1, Height, Width)
		self.images = tt.from_numpy(self.images)  # Convert to tensor

		self.labels = tt.from_numpy(labels)  # (Batch)
		self.labels = tt.nn.functional.one_hot(self.labels.long(), num_classes=10)  # (Batch, 10)

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index): # get image-label pair
		return dict(image=self.images[index], label=self.labels[index])

# if __name__ == "__main__":