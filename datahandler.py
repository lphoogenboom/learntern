from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf

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

	def csvLoadMNIST(self, csvPath): # load both CSVs and merge
		targetPath = Path(__file__).resolve().parent
		path = csvPath.split("/")
		print(path)
		for dir in path:
			targetPath = targetPath/dir
		data = pd.read_csv(targetPath)
		images = data.drop(columns=['label']).values
		images = images.reshape(images.shape[0],28,28).astype(np.uint8)

		labels = data['label'].values.astype(np.uint8) # todo: --> 1-hot-encoding e.g. [0 1 0 ... 0] = 1
		return [images, labels]


if __name__ == "__main__":
	print("==== RAN AS FILE ====")
	loader = Dataloader()

	loader.unzipMNIST('data/raw/mnist.zip')

	# images_train, labels_train = loader.csvLoadMNIST('data/raw/mnist_test.csv')


	


