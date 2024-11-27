import torch.nn as nn
import torch.nn.functional as ff

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.perceptron = nn.Linear(1, 1, bias=True)

    def forward(self, input):
        output = self.perceptron(input)
        return output


if __name__ == "__main__":
    print("==== RAN AS FILE ====")
    # net = Model()
    # data = dataset(images,labels)
    # output = net.forward(data)
    # print(output)  # 3x10 tensor
    # print(output.sum(dim=1))  # 3x1 tensor with values of 1 (sum of probabilities is 1)