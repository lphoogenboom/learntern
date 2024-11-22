from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf
import torch as tt

class Dataloader():

	def unzipMNIST(self, zipPath): # unzip archived files
		if Path('data/csv/mnist_test.csv').is_file():
			print('At least 1 unzipped file exist... \nRemove file or change names and try again.')
			return

		basePath = Path(__file__).resolve().parent # present working directory
		relativePath = zipPath.split("/") # path from pwd to .zip (including file)

		# append relative path to base path
		targetPath = basePath
		for dir in relativePath: 
			targetPath = targetPath/dir

		with zf.ZipFile(targetPath, 'r') as archive:
			archive.extractall('data/csv')

	def csvLoadMNIST(self, csvPath): # load csv and seperate images from labels
		targetPath = Path(__file__).resolve().parent
		path = csvPath.split("/")
		for dir in path:
			targetPath = targetPath/dir
		data = pd.read_csv(targetPath)
		images = data.drop(columns=['label']).values
		images = images.reshape(images.shape[0],28,28).astype(np.uint8)

		labels = data['label'].values.astype(np.uint8) # todo: --> 1-hot-encoding e.g. [0 1 0 ... 0] = 1
		return [images, labels]
	
class Dataset(tt.utils.data.Dataset):

	def __init__(self,images,labels):
		self.images = images # [Batch, Height, Width]
		self.images = self.images[:, None]
		self.images = tt.from_numpy(self.images)

		self.labels = tt.from_numpy(labels) # [Batch]
		self.labels = tt.nn.functional.one_hot(self.labels.long(),num_classes=10) # [Batch, Label]

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index): # get image-label pair
		return dict(image=self.images[index], label=self.labels[index])

if __name__ == "__main__":
	print("==== RAN AS FILE ====")
	loader = Dataloader()
	# loader.unzipMNIST('data/raw/mnist.zip')
	images_train, labels_train = loader.csvLoadMNIST('data/csv/mnist_test.csv') # images is 1000x28x28

	here = Path(__file__).resolve().parent
	path_splits = here/"data"/"splits"
	path_csv = here/"data"/"csv"
	path_train = path_csv/"mnist_train.csv"
	path_test = path_csv/"mnist_test.csv"

	data_train = pd.read_csv(path_train)
	data_test = pd.read_csv(path_test)
	data_full = pd.concat([data_train, data_test], ignore_index=True)

	labels = data_full['label'].values.astype(np.uint8)
	images = data_full.drop(columns=['label']).values
	images = images.reshape(images.shape[0],28,28).astype(np.uint8) # create tensor for tt

	np.save(path_processed/"images.npy", images)
	np.save(path_processed/"labels.npy", labels)