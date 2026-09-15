import torch
import torch.nn as nn

class CustomDriveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Dla Tanh (Skręt: [-1, 1]) - odporny na szumy i zapobiega gwałtownym ruchom
        self.steering_loss = nn.SmoothL1Loss()
        
        # 2. Dla Sigmoid (Gaz i Hamulec: [0, 1]) - szybko uczy wciskania 0.0 lub 1.0

    def forward(self, predictions, targets):
        # predictions i targets mają kształt [batch_size, 3]
        # index 0: skręt (tanh)
        # index 1: gaz (sigmoid)
        # index 2: hamulec (sigmoid)
        
        loss_steer = 2 * self.steering_loss(predictions[:, 0], targets[:, 0])
        loss_first = self.steering_loss(predictions[:, 1], targets[:, 1])
        loss_second =  self.steering_loss(predictions[:, 2], targets[:, 2])
        # Całkowity loss
        return loss_steer + loss_first + loss_second