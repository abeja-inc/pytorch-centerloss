import math
import torch
import torch.nn as nn


class EMA(nn.Module):
    def __init__(self, size, alpha, requires_grad=False, eps=1.0e-6):
        super(EMA, self).__init__()
        assert(0 < alpha and alpha < 1)
        self.centers = nn.Parameter(torch.zeros(*size), requires_grad=False)
        self.counts = nn.Parameter(torch.zeros(size[0]), requires_grad=False)
        self.alpha = alpha
        self.log_alpha = math.log(alpha)
        self.eps = eps

    def forward(self, i, x):
        if self.training:
            # update center and counter
            self.centers.data[i] -= (1 - self.alpha) * (self.centers[i].data - x.data)
            self.counts.data[i] += 1
        center = self.centers[i]
        c = 1 - torch.exp(self.log_alpha * self.counts[i])  # 1 - alpha ^ count
        return center / c.unsqueeze(1).expand_as(center)
