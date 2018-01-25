import torch
import torch.nn as nn

__all__ = ['CenterLoss', 'ImprovedCenterLoss']

class CenterLoss(nn.Module):
    def __init__(self, num_classes, dim, alpha=0.95):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.centers = nn.Parameter(torch.zeros(num_classes, dim), requires_grad=False)
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, x, y):
        center = self.centers[y.data]
        loss = self.mse_loss(x, center)
        if self.training:
            self.centers.data[y.data] -= (1 - self.alpha) * (center.data - x.data)
        return loss

class ImprovedCenterLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.95):
        super(ImprovedCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.centers = nn.Parameter(torch.zeros(num_classes, num_classes),
                                    requires_grad=False)
        self.mse_loss = nn.MSELoss()
        self.alpha = alpha

    def forward(self, x, y):
        center = self.centers[y.data]
        loss = self.mse_loss(x, center)
        if self.training:
            self.centers.data[y.data] -= (1 - self.alpha) * (center.data - x.data)
        return loss
    
