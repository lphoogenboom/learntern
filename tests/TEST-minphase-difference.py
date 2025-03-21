import torch as tt
import torch.nn as nn
import torch.nn.functional as ff

angle1 =60.0
angle2 = -20.0

angle_difference = abs(angle1-angle2)
print(angle_difference)

corrected_angle_difference = (angle_difference + 180) % 360 - 180
print(corrected_angle_difference)

loss_rotation = ff.l1_loss(tt.tensor(0), tt.tensor(corrected_angle_difference))
print(loss_rotation)