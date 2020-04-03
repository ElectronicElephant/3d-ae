import json
from typing import Tuple

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from torch import FloatTensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import NumpyVoxelDataset, PickerDataset
from visualizer import plot


class DenoisingAutoEncoder(pl.LightningModule):
    logger: CometLogger

    def __init__(self, hparams, data_path, array_name):
        super().__init__()

        self.hparams = hparams
        self.vector_length = self.hparams["vector_length"]
        self.dim = self.hparams["dim"]
        self.lr = self.hparams["lr"]
        self.bs = self.hparams["batch_size"]
        self.normal_mean = self.hparams["normal_mean"]
        self.normal_sigma = self.hparams["normal_sigma"]
        self.threshold = self.hparams["threshold"]

        self.encoder = nn.Sequential(
            nn.Conv3d(1, 15, 2, 2),
            nn.LeakyReLU(),
            nn.Conv3d(15, 50, 2, 2),
            nn.LeakyReLU(),
            nn.Conv3d(50, 100, 2, 2),
            nn.LeakyReLU(),
        )
        # TODO: INPUT SIZE and Length! as hyper-parameters
        self.fc1 = nn.Linear(100 * 4 * 4 * 4, self.vector_length)
        self.fc2 = nn.Linear(self.vector_length, 100 * 4 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(100, 50, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(50, 15, 2, 2),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(15, 1, 2, 2),
            nn.Sigmoid()
        )

        # Data
        self.train_dataset = NumpyVoxelDataset(data_path, self.dim, array_name)
        self.val_dataset = PickerDataset(self.train_dataset, ids=[10, 513, 10403, 20303, 40013])

    def forward(self, x):
        feature = self.encoder(x)  # 32, 100, 2, 2, 2
        z = self.fc1(feature.view(-1, 100 * 4 * 4 * 4))  # TODO: 800 - INPUT SIZE
        feature2 = F.relu(self.fc2(z))
        feature2 = feature2.view(-1, 100, 4, 4, 4)  # B, C, D, H, W  TODO: INPUT SIZE
        recon_x = self.decoder(feature2)
        return recon_x, z

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=len(self.val_dataset), shuffle=False, pin_memory=True)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(),
        #                        lr=0.001, momentum=0.99, weight_decay=0.0005)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, image: FloatTensor, noise: bool = True) -> Tuple[FloatTensor, FloatTensor]:
        if noise:
            # noinspection PyArgumentList
            gauss = FloatTensor(*image.shape).to(image.device).normal_(mean=self.normal_mean, std=self.normal_sigma)
            noised = image + gauss
        else:
            noised = image

        # TODO: Check Shape
        recon, _ = self(noised)

        return image, recon

    def training_step(self, batch, batch_idx):
        image, recon = self._step(batch)

        loss = F.binary_cross_entropy(recon, image, reduction='mean', weight=None)

        _max = recon.max()
        logs = {'loss': loss, 'max': _max}
        return {**logs, "log": logs}

    def validation_step(self, batch, batch_idx):
        image, recon = self._step(batch, noise=False)

        loss = F.binary_cross_entropy(recon, image, reduction='mean', weight=None)

        srcs = image.detach().cpu().numpy()
        dests = recon.detach().cpu().numpy()

        for idx, (src, dest) in enumerate(zip(srcs, dests)):
            src.resize(([self.dim] * 3))
            dest.resize(([self.dim] * 3))
            plt.close()
            plot(src, dest > self.threshold)
            self.logger.experiment.log_figure(step=self.current_epoch, figure_name=f"figure_{idx}")

        logs = {"val_loss": loss}
        return {**logs, "log": logs}


if __name__ == '__main__':
    with open("tokens.json", mode="r") as f:
        conf = json.load(f)
    comet_logger = CometLogger(**conf["comet"])
    hparams = {
        "dim": 32,
        "vector_length": 128,
        "normal_mean": 0,
        "normal_sigma": 0.05,
        "lr": 1e-3,
        "batch_size": 128,
        "threshold": 0.3
    }
    model = DenoisingAutoEncoder(**conf["data"], hparams=hparams)
    # model = DenoisingAutoEncoder(hparams=hparams, data_path='voxel.npz', array_name="arr_0")
    trainer = Trainer(gpus=1, logger=comet_logger)
    trainer.fit(model)
