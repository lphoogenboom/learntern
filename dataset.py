from pathlib import Path
from dataProcessor import dataProcessor
import numpy as np
import torch as tt
import sys

# For datasplit


class Dataset(tt.utils.data.Dataset):

	def __init__(self,input,output):
		self.x = input  # (Batch, Height, Width)
		self.x = self.x[:, None]  # (Batch, 1, Height, Width) required by torch
		self.x = tt.from_numpy(self.x)  # Convert to tensor

		self.y = tt.from_numpy(output)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, index): # get image-label pair
		return dict(x=self.x[index], y=self.y[index])

if __name__ == "__main__":
	print("==== RAN AS FILE ====")
	processor = dataProcessor()
	pwd = processor.getParentDir()
	data_train = np.load(pwd/"data"/"arrays"/"linearTrain.npy")
	data_test = np.load(pwd/"data"/"arrays"/"linearTest.npy")

	data_train_torch = Dataset(data_train[:,0],data_train[:,1])

	
	