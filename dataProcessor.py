from pathlib import Path
import pandas as pd
import numpy as np
import zipfile as zf
import torch as tt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

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
    
    def trainTestSplit(self,labels):
        datasplit = dict()
        idx_train, idx_test, _, _ = train_test_split(
            range(len(labels)),
            labels,
            test_size=0.2,
            stratify=labels,
        )
        datasplit["train"] = idx_train
        datasplit["test"] = idx_test
        return datasplit

    def kFoldSplit(self, data, indices, key, k):
        skf_train_val = StratifiedKFold(n_splits=k, shuffle=True)
        for i, (idx_train, idx_va) in enumerate(skf_train_val.split(indices[key], data[indices[key]])):
            indices[f"train_{i:02d}"] = idx_train
            indices[f"val_{i:02d}"] = idx_va
        return indices
    
if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    
    ''' For Saving the data as numpy arrays '''
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

    ''' For Datasplitting '''
    processor = dataProcessor()
    here = processor.getParentDir()
    path_data = here/"data"/"arrays" # Original Data
    path_splits = here/"data"/"splits" # Destination for splits

    ## Load Data Arrays
    images = np.load(path_data/"images.npy")
    labels = np.load(path_data/"labels.npy")

    # # split contains test data + train data + train data-->(5 * (train_##,val_##) overlapping)
    # split = np.load(path_splits/"datasplit.npz")

    ## Test Train Split
    dataset = Dataset(images,labels)
    split_tt = dataset.trainTestSplit(labels) # Named columns "train"/"test"

    # print((labels.keys()))

    ## split training data 5-fold
    split_5fold = dataset.kFoldSplit(labels, split_tt, "train", 5) # includes test data

    ## Save Splits
    np.savez(path_splits/"datasetIndices.npz", **split_5fold)