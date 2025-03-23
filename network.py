import torch.nn as nn
import torch as tt
from data import DataAugmenter

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmenter = DataAugmenter()

        self.localisation_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 28 -> 14

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2), # 14 -> 7

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=1) # 7 -> 5
        )

        self.localisation_dense = nn.Sequential(
            nn.Linear(128*5*5,128),
            nn.Linear(128,64),
            nn.Linear(64,6),
        )

        self.localisation_dense[-1].weight.data.zero_()
        self.localisation_dense[-1].bias.data.copy_(tt.tensor([1, 0, 0, 1, 0, 0], dtype=tt.float))

    def forward(self, image):
        # run image through spatial transformer
        feature = self.localisation_conv(image)
        feature_flat = feature.view(feature.size(0), -1)  # Flatten for fully connected layer
        theta = self.localisation_dense(feature_flat)
        transform = tt.reshape(theta, (theta.size(0),2,3))



        return theta, transform# return corrected image and correction transform