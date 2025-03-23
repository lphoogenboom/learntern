import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import numpy as np

class Criterion(nn.Module):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

	def forward(self, estimate, reference):
		loss_1 = ff.smooth_l1_loss(estimate['theta_1'], reference['theta_1'])
		loss_2 = ff.smooth_l1_loss(estimate['theta_2'], reference['theta_2'])
		loss_3 = ff.smooth_l1_loss(estimate['theta_3'], reference['theta_3'])
		loss_4 = ff.smooth_l1_loss(estimate['theta_4'], reference['theta_4'])
		loss_5 = ff.smooth_l1_loss(estimate['theta_5'], reference['theta_5'])
		loss_6 = ff.smooth_l1_loss(estimate['theta_6'], reference['theta_6'])
		loss_l1 = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6
		
		# Maybe regularisation helps?
		regularisation_loss = tt.mean(( # MSE with identity transform
						estimate['theta_1'] - 1)**2 + 
                       (estimate['theta_2'] - 0)**2 +  
                       (estimate['theta_3'] - 0)**2 +  
                       (estimate['theta_4'] - 0)**2 + 
                       (estimate['theta_5'] - 1)**2 +  
                       (estimate['theta_6'] - 0)**2)   

		weight_regularisaiton = 1e-4  # Start small, adjust later
		loss_total = loss_l1 + weight_regularisaiton * regularisation_loss
		return loss_total