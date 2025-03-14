import torch.nn as nn
import torch as tt
import torch.nn.functional as ff
import numpy as np
from data import dataProcessor
from dataset import Dataset

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        ## Define the layers
        # e.g.:
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
        )
        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.dense = nn.Linear(144, 10)
        self.maxpool_01 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.maxpool_02 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.final_activation = nn.Softmax(dim=1)

    def forward(self, input):
        hidden = self.conv_block_01(input)     # 28 -> 28
        hidden = self.maxpool_01(hidden)       # 28 ->  7

        hidden = self.conv_block_02(hidden)    # 13 ->   13
        hidden = self.maxpool_02(hidden)       # 7 -> 3.5 (-> 3)
        hidden = hidden.view(hidden.size(0), -1)  # Flatten for fully connected layer
        # 32 channels * (3x3) maps => 288 Neurons
        hidden = self.dense(hidden)  # Fully connected layer
        # Fully connected 288->10
        out = self.final_activation(hidden)  # Softmax activation
        # maps to normalised vector
        return out


if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    processor = dataProcessor()
    pwd = processor.getParentDir()
    images = np.load(pwd/"data"/"arrays"/"images.npy")
    labels = np.load(pwd/"data"/"arrays"/"labels.npy")
    splits = np.load(pwd/"data"/"splits"/"5-fold-indices.npz")
    data = tt.rand(1, 1, 28, 28)

    net = Model()
    output = net.forward(data)
    print(output)

    # data = dataset(images,labels)
    # output = net.forward(data)
    # print(output)  # 3x10 tensor
    # print(output.sum(dim=1))  # 3x1 tensor with values of 1 (sum of probabilities is 1)