import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        ## Define the layers
        # e.g.:
        self.conv_block_00 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.conv_block_01 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.conv_block_02 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.conv_block_03 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # self.conv_block_04 = nn.Sequential(
        #     nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        # )

        self.dense1 = nn.Linear(512*5*5, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.output = nn.Linear(128,4)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=1)
        # self.final_activation = nn.Softmax(dim=1)

    def forward(self, input):
        hidden = self.conv_block_00(input)   # 28 -> 28 
        hidden = self.maxpool(hidden)        # 28 ->  14
        
        hidden = self.conv_block_01(hidden)  # 14 -> 14
        hidden = self.maxpool(hidden)        # 14 -> 7        

        hidden = self.conv_block_02(hidden)  # 7 -> 7
        hidden = self.maxpool2(hidden)       # 7->5

        hidden = self.conv_block_03(hidden)  # 5 -> 5

        hidden = hidden.view(hidden.size(0), -1)  # Flatten for fully connected layer
        hidden = self.dense1(hidden)
        hidden = self.dense2(hidden)
        hidden = self.dense3(hidden)
        
        out = self.output(hidden)
        return out