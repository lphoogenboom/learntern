from pathlib import Path
import pandas as pd
import numpy as np

class Dataloader():

	def csvLoadMNIST(self, csvPath):
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
	images_train, labels_train = loader.csvLoadMNIST('data/raw/mnist_test.csv')


	


