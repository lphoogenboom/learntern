import numpy as np
import pandas as pd
from dataset import dataset
import dataProcessor
from sklearn.model_selection import train_test_split
import torch as tt
from network import Model


'''Define Learning Step'''
def step(phase):
    # Prepare model for training or evaluation
    if phase == "train":
        model.train()
        tt.set_grad_enabled(True)
    else:
        model.eval()
        tt.set_grad_enabled(False)

    # Initialize loss and time
    loss_step = 0.0
    time_step = time()

    # Iterate over data loader
    accuracy = [0, 0]

    for data in dl[phase]:
        # Move data to device
        image = data["image"].to(device)
        label = data["label"].to(device)

        # Get predictions
        prediction = model(image)

        # Calculate loss
        loss = criterion(prediction, label)

        # Update running loss
        loss_step += loss.item()

        if phase == "train":
            # Backward pass
            loss.backward()

            # Update model
            optimizer.step()
            optimizer.zero_grad()

        accuracy[0] += tt.sum(tt.argmax(prediction, dim=1) == tt.argmax(label, dim=1)).item()
        accuracy[1] += len(label)

    # Log step
    loss_step = loss_step / len(dl[phase])
    log[f"loss_{phase}"].append(loss_step)
    log[f"accuracy_{phase}"].append(accuracy[0] / accuracy[1])
    log[f"time_{phase}"].append(time() - time_step)


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

'''Save data in torch-ready format'''
x_train = data['x'][idx_train].to_numpy().reshape((len(idx_train),1))
y_train = data['y'][idx_train].to_numpy().reshape((len(idx_train),1))
data_train = np.concatenate([x_train,y_train],axis=1)
np.save('./data/arrays/linearTrain.npy', data_train)


x_test = data['x'][idx_test].to_numpy().reshape((len(idx_test),1))
y_test = data['y'][idx_test].to_numpy().reshape((len(idx_test),1))
data_test = np.concatenate([x_test,y_test],axis=1)
np.save('./data/arrays/linearTest.npy', data_test)

'''Create Dataset'''
pwd = processor.getParentDir()
data_train = np.load(pwd/"data"/"arrays"/"linearTrain.npy")
data_test = np.load(pwd/"data"/"arrays"/"linearTest.npy")

data_train_torch = dataset(data_train[:,0],data_train[:,1])
data_test_torch = dataset(data_test[:,0],data_test[:,1])

'''Use dataloader to pass data to torch'''
dl_train = tt.utils.data.DataLoader(data_train_torch, batch_size=5, shuffle=True)

'''Instantiate model'''
model = Model()

'''learn model'''

'''test model'''

'''log, visualise, etc'''
