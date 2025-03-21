import torch as tt
import torch.nn as nn
import torch.nn.functional as ff
import numpy as np

class Criterion(nn.Module):

	def __init__(self, *args, **kwargs) -> None:
		super().__init__(*args, **kwargs)

	def forward(self, estimate, reference):
		minimum_phase_angle_estimate = (estimate['rotation'] + np.pi) % (2*np.pi) - np.pi
		right_phase_difference = reference['rotation']-minimum_phase_angle_estimate
		left_phase_difference = tt.sign(reference['rotation'])*(np.pi-tt.abs(reference['rotation']))-tt.sign(minimum_phase_angle_estimate)*(np.pi-tt.abs(minimum_phase_angle_estimate))
		base_angle = tt.tensor(np.zeros(200)).to(tt.float32).to("mps")
		phase_differences = tt.stack([left_phase_difference,right_phase_difference],dim=1)
		
		_,idx = tt.min(tt.abs(phase_differences),dim=1,keepdim=False)
		minimum_phase_difference = phase_differences[tt.arange(200),idx]
		# idx = tt.argmax(tt.abs(minimum_phase_difference))
		# print(
		# 	f"Reference: {reference['rotation'][idx].item():13.10f} | "
 	    #  	f"Estimate: {estimate['rotation'][idx].item():13.10f} | "
      	# 	f"Difference: {minimum_phase_difference[idx].item()}")

		loss_rotation = ff.l1_loss(base_angle, minimum_phase_difference)
		loss_translation_x = ff.l1_loss(estimate['translation'][:,0], reference['translation'][:,0])
		loss_translation_y = ff.l1_loss(estimate['translation'][:,1], reference['translation'][:,1])
		loss_scale = ff.l1_loss(estimate['scale'], reference['scale'])
		loss = loss_rotation + loss_translation_x + loss_translation_y + loss_scale
		return loss