"""
The main structure is adopted from STS (https://arxiv.org/abs/1611.07932)
The original autoencoder is changed to denoising ae.

The structure of this code is adapted from pytorch official tutorial https://github.com/pytorch/examples/blob/master/vae/main.py
"""

from torch import nn
from torch.nn import functional as F


class DAE(nn.Module):
    def __init__(self, vector_length, bottle_neck=True):
        super(DAE, self).__init__()
        self.bottle_neck = bottle_neck
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 * 32
            nn.Conv2d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16 * 16
            nn.Conv2d(20, 50, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8 * 8
            nn.Conv2d(50, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 4 * 4
        if self.bottle_neck:
            self.bottle_down = nn.Sequential(
                nn.Conv2d(100, 200, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 2 * 2
                nn.Conv2d(200, 400, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2),  # 1 * 1
                nn.Conv2d(400, vector_length, 1, 1, 0))
            self.bottle_up = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(vector_length, 400, 1, 1, 0),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(400, 200, 3, 1, 1),  # 2 * 2
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.ReLU())
        self.fc1 = nn.Linear(1600, vector_length)
        self.fc2 = nn.Linear(vector_length, 1600)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(100, 50, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(50, 20, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(20, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(10, 1, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        print(f"input x {x.size()}")
        feature = self.encoder(x)
        print(f"feature {feature.size()}")
        if self.bottle_neck:
            print("Bottle_neck")
            z_out = self.bottle_down(feature)
            z = z_out  # need transpose
            feature2 = self.bottle_up(z_out)
        else:
            print("NO Bottle_neck")
            z = self.fc1(feature.view(-1, 1600))
            feature2 = F.relu(self.fc2(z))
            feature2 = feature2.view(-1, 100, 4, 4)  # B, C, H, W
        recon_x = self.decoder(feature2)
        return recon_x, z

