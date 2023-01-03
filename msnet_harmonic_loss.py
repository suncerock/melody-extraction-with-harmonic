import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


class MSNet(nn.Module):
    def __init__(self, device):
        super(MSNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
        )
        self.pool1 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
        )
        self.pool2 = nn.MaxPool2d((4, 1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
        )
        self.pool3 = nn.MaxPool2d((4, 1), return_indices=True)

        self.up_pool3 = nn.MaxUnpool2d((4, 1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
        )
        self.up_pool2 = nn.MaxUnpool2d((4, 1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
        )

        self.up_pool1 = nn.MaxUnpool2d((4, 1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 4, 5, padding=2),
            nn.SELU()
        )
        
        self.device = device
        self.to(device)
        
        # self.loss_fn = HarmonicLoss()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch, requires_loss=True):
        x, y_melody, y_harmonic, y_subharmonic = batch
        x = x.to(self.device)
        # y_melody = y_melody.to(self.device)
        # y_harmonic = y_harmonic.to(self.device)
        # y_subharmonic = y_subharmonic.to(self.device)
        
        target = torch.zeros_like(y_subharmonic).long()
        target[y_subharmonic == 1] = 3
        target[y_harmonic == 1] = 2
        target[y_melody == 1] = 1
        
        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))
        
        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))
        
        with torch.no_grad():
            output = torch.softmax(u1, dim=1)
            output = torch.softmax(output[:, 1], dim=1)

        if requires_loss:
            loss = self.loss_fn(u1, target.to(self.device))
            return loss, output
        else:
            return output
