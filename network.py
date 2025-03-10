import torch.nn as nn
import torch as tt
import torch.nn.functional as ff
import numpy as np
from data import DataManager
from dataset import Dataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        ## Define the layers
        # e.g.:
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )

        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        self.conv_block_03 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.dense1 = nn.Linear(576, 1)
        self.dense2 = nn.Linear(360, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.final_activation = nn.Softmax(dim=1)

    def forward(self, input):
        hidden = self.conv_block_00(input)   # 28 -> 28 
        hidden = self.conv_block_01(hidden)   # 28 -> 28
        hidden = self.maxpool(hidden)        # 28 ->  14

        hidden = self.conv_block_02(hidden)  # 14 -> 14
        hidden = self.maxpool(hidden)        # 14 ->  7

        hidden = self.conv_block_03(hidden)  #  7 ->  7
        hidden = self.maxpool(hidden)        #  7 ->  3.5 (-> 3)

        hidden = hidden.view(hidden.size(0), -1)  # Flatten for fully connected layer
        out = 360*self.dense1(hidden)
        # hidden = tt.sigmoid(hidden)
        # hidden = self.dense2(hidden)
        # out = 360*tt.sigmoid(hidden)
        # Fully connected 288->10
        # out = self.final_activation(hidden)  # Softmax activation
        # maps to normalised vector
        return out


if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    processor = DataManager()
    pwd = processor.getParentDir()
    images = np.load(pwd/"data"/"arrays"/"images.npy")
    labels = np.load(pwd/"data"/"arrays"/"labels.npy")
    splits = np.load(pwd/"data"/"splits"/"5-fold-indices.npz")
    data = tt.rand(2, 1, 28, 28)
    print(data.shape)
    print(type(data))

    net = Model()
    output = net.forward(data)
    print(output)

    # data = dataset(images,labels)
    # output = net.forward(data)
    # print(output)  # 3x10 tensor
    # print(output.sum(dim=1))  # 3x1 tensor with values of 1 (sum of probabilities is 1)