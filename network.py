import torch.nn as nn
import torch.nn.functional as ff

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        ## Define the layers
        # e.g.:
        # self.block_0 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.etc...
        # )

    def forward(self, input):
        # hidden = self.block_0(input)
        # out = self.block_1(hidden)
        # return out
        return 1 # placeholder


if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    # net = Model()
    # data = dataset(images,labels)
    # output = net.forward(data)
    # print(output)  # 3x10 tensor
    # print(output.sum(dim=1))  # 3x1 tensor with values of 1 (sum of probabilities is 1)