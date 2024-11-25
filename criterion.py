import torch as tt
import torch.nn as nn
import torch.functional as ff

class Criterion(nn.Module):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

	def forward(self, prediction, reference):
		loss = ff.l1_loss(prediction, reference)
		return loss