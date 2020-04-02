import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import DAENumpyVoxelDataset


class DenoisingAutoEncoder(pl.LightningModule):
    def __init__(self, vector_length, dim, data_path, array_name):
        super(DenoisingAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),  # TODO: Try leaky ReLu
            nn.MaxPool3d(2),  # 1 / 2
            nn.Conv3d(10, 20, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 1 / 4
            nn.Conv3d(20, 50, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2),  # 1 / 8
            nn.Conv3d(50, 100, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool3d(2)  # 1 / 16
        )
        # TODO: INPUT SIZE and Length! as hyper-parameters
        self.fc1 = nn.Linear(800, vector_length)
        self.fc2 = nn.Linear(vector_length, 800)
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
            nn.Sigmoid()
        )

        # Data
        self.train_dataset = DAENumpyVoxelDataset(data_path, dim, array_name)

    def forward(self, x):
        feature = self.encoder(x)  # 32, 100, 2, 2, 2
        z = self.fc1(feature.view(-1, 800))  # TODO: 800 - INPUT SIZE
        feature2 = F.relu(self.fc2(z))
        feature2 = feature2.view(-1, 100, 2, 2, 2)  # B, C, D, H, W  TODO: INPUT SIZE
        recon_x = self.decoder(feature2)
        return recon_x, z

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=64,
                          shuffle=True,
                          num_workers=64)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(),
        #                        lr=0.001, momentum=0.99, weight_decay=0.0005)
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        image, noised = batch
        # TODO: Check Shape
        recon, _ = self(noised)

        loss = F.binary_cross_entropy(recon, image, reduction='mean', weight=None)

        return {'BCE': loss}


if __name__ == '__main__':
    model = DenoisingAutoEncoder(data_path='voxels.npz', array_name='arr_0.npy',
                                 dim=32, vector_length=128)
    trainer = Trainer(gpus=1)
    trainer.fit(model)
