import torch as tt
import torch.nn as nn
import torch.nn.functional as ff

class Criterion(nn.Module):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

	def forward(self, prediction, reference):
		
		loss = ff.l1_loss(prediction, reference)
		return loss
	
if __name__ == "__main__":
	crit = Criterion()
	input = tt.randn(2, 10, requires_grad=True)
	target = tt.randn(2, 10)
	output = crit.forward(input, target)
	print(output)

	output.backward()