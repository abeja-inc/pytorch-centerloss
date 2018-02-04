import torch
import torch.nn as nn

class Conv2dBlock(nn.Module):
    
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, leak=0.1):
        super(Conv2dBlock, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.leak = leak

        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(leak),
        )
        
    def forward(self, x):
        return self.main(x)
    

class ConvLarge(nn.Module):

    def __init__(self):
        super(ConvLarge, self).__init__()

        self.net = nn.Sequential(
            Conv2dBlock(3, 128, 3, 1, 1),
            Conv2dBlock(128, 128, 3, 1, 1),
            Conv2dBlock(128, 128, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.5),
            Conv2dBlock(128, 256, 3, 1, 1),
            Conv2dBlock(256, 256, 3, 1, 1),
            Conv2dBlock(256, 256, 3, 1, 1),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout2d(0.5),
            Conv2dBlock(256, 512, 3, 1, 1),
            Conv2dBlock(512, 256, 1, 1, 0),
            Conv2dBlock(256, 128, 1, 1, 0),
            nn.AvgPool2d(8, 1),
        )
        
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 3, 32, 32)
        x = self.net(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x  # return logit
