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
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # TODO: Try leaky ReLu
            nn.MaxPool3d(2),  # 32 * 32
            nn.Conv3d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 16 * 16
            nn.Conv3d(20, 50, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 8 * 8
            nn.Conv3d(50, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2))  # 4 * 4
        if self.bottle_neck:
            self.bottle_down = nn.Sequential(
                nn.Conv3d(100, 200, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 2 * 2
                nn.Conv3d(200, 400, 3, 1, 1),
                nn.ReLU(),
                nn.MaxPool3d(2),  # 1 * 1
                nn.Conv3d(400, vector_length, 1, 1, 0))
            self.bottle_up = nn.Sequential(
                nn.ReLU(),
                nn.Conv3d(vector_length, 400, 1, 1, 0),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(400, 200, 3, 1, 1),  # 2 * 2
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.ReLU())
        self.fc1 = nn.Linear(1024, vector_length)
        self.fc2 = nn.Linear(vector_length, 1024)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(100, 50, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(50, 20, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(20, 10, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='trilinear'),
            nn.Conv3d(10, 1, 3, 1, 1),
            nn.Sigmoid())

    def forward(self, x):
        feature = self.encoder(x)
        if self.bottle_neck:
            z_out = self.bottle_down(feature)
            z = z_out  # need transpose
            feature2 = self.bottle_up(z_out)
        else:
            z = self.fc1(feature.view(-1, 1024))  # TODO: decrease the value and move to config
            feature2 = F.relu(self.fc2(z))
            feature2 = feature2.view(-1, 16, 4, 4, 4)  # B, C, D, H, W
        recon_x = self.decoder(feature2)
        return recon_x, z

