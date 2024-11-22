from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf
import torch as tt

class dataProcessor():

    def getParentDir(self):
        parent_dir = Path(__file__).resolve().parent
        return parent_dir

    def unzipData(self, zipPath): # unzip archived files
        if Path('data/csv/mnist_test.csv').is_file():
            print('At least 1 unzipped file exist... \nWill skip unzip until target files are removed.')
            return
        
        basePath = self.getParentDir()
        relativePath = zipPath.split("/") # path from pwd to .zip (including file)

        # append relative path to base path
        targetPath = basePath
        for dir in relativePath: 
            targetPath = targetPath/dir

        with zf.ZipFile(targetPath, 'r') as archive:
            archive.extractall('data/csv')

    def csvLoad(self, csvPath): # load csv and seperate images from labels
        targetPath = self.getParentDir()
        path = csvPath.split("/")
        for dir in path:
            targetPath = targetPath/dir
        data = pd.read_csv(targetPath)
        return data
    
    def writeDataArray(self, data):
        return
    
if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    processor  = dataProcessor()
    processor.unzipData("data/archived/mnist.zip")
    data_train = processor.csvLoad("data/csv/mnist_train.csv")
    data_test  = processor.csvLoad("data/csv/mnist_test.csv")
    data_full  = pd.concat([data_train, data_test], ignore_index=True)
    
    labels = data_full['label'].values.astype(np.uint8)
    images = data_full.drop(columns=['label']).values
    images = images.reshape(images.shape[0],28,28).astype(np.uint8) # create tensor for tt

    path_numpy = processor.getParentDir()/"data"/"arrays"

    np.save(path_numpy/"images.npy", images)
    np.save(path_numpy/"labels.npy", labels)