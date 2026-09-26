"""
Plik: LossFunc.py
Opis: Zawiera definicję niestandardowej funkcji kosztu (CustomDriveLoss) 
      używanej podczas trenowania modelu. Funkcja ta waży błędy sterowania, 
      gazem i hamulcem, aby zapewnić bardziej precyzyjne uczenie się zachowań kierowcy.
"""
import torch
import torch.nn as nn

class CustomDriveLoss(nn.Module):
    def __init__(self,alpha,power):
        super().__init__()
        # 1. Dla Tanh (Skręt: [-1, 1]) - odporny na szumy i zapobiega gwałtownym ruchom
        self.steering_loss = nn.SmoothL1Loss()
        self.alpha = alpha
        self.pow = power
        
        # 2. Dla Sigmoid (Gaz i Hamulec: [0, 1]) - szybko uczy wciskania 0.0 lub 1.0

    def forward(self, predictions, targets):
        # predictions i targets mają kształt [batch_size, 3]
        # index 0: skręt (tanh)
        # index 1: gaz (sigmoid)
        # index 2: hamulec (sigmoid)
        error = torch.abs(predictions-targets)
        weight = 1.0 + self.alpha * torch.abs(targets).pow(self.pow)
        # loss_steer = 2 * self.steering_loss(predictions[:, 0], targets[:, 0])
        # loss_first = self.steering_loss(predictions[:, 1], targets[:, 1])
        # loss_second =  self.steering_loss(predictions[:, 2], targets[:, 2])
        # Całkowity loss
        loss = weight*error
        return loss.mean()