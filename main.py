import pandas as pd
import dataset
import dataProcessor
from sklearn.model_selection import train_test_split


'''We need to import the dataframe containing the perturbed data'''
data = pd.read_pickle('./data/pickles/linearData.pkl')

'''We now need to turn this data into a dataset for torch'''
processor = dataProcessor.dataProcessor()
# print(data['x'])

idx_train, idx_test, _, _ = train_test_split(
            range(len(data['y'])),
            data['y'],
            test_size=0.2,
            # stratify=data['y'], data is distributed well anough to not bother
        )
