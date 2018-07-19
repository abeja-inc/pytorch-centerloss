import math
import torch
import torch.nn as nn


class EMA(nn.Module):
    
    def __init__(self, size, alpha):
        super(EMA, self).__init__()
        assert(0 < alpha and alpha < 1)
        self.centers = torch.zeros(*size)
        self.counts = torch.zeros(*size)
        self.alpha = alpha
        self.log_alpha = math.log(alpha)

    def forward(self, i, x):
        if self.training:
            # update center and counter
            self.centers[i] -= (1 - self.alpha) * (self.centers[i] - x)
            self.counts[i] += 1
        center = self.centers[i]
        c = 1 - torch.exp(self.log_alpha * self.counts[i])  # 1 - alpha ^ count
        return center / c

    def to(self, device):
        self.centers = self.centers.to(device)
        self.counts = self.counts.to(device)
