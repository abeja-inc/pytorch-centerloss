import math
import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, size, alpha, requires_grad=False, eps=1.0e-6):
        super(EMA, self).__init__()
        self.centers = nn.Parameter(torch.zeros(*size), requires_grad=False)
        self.counts = nn.Parameter(torch.zeros(size[0]), requires_grad=False)
        self.alpha = alpha
        self.log_alpha = math.log(alpha)
        self.eps = eps

    def forward(self, i, x):
        center = self.centers[i].clone()
        counts = self.counts[i]
        c = 1 + self.eps  - torch.exp(self.log_alpha * counts)  # 1 - alpha^counts
        center /= (c.unsqueeze(1).expand_as(center))
        if self.training:
            self.centers.data[i] -= (1 - self.alpha) * (center.data - x.data)
            self.counts.data[i] += 1            
        return center
