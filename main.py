import datetime
import json
import operator
import os
from argparse import Namespace
from functools import reduce
from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CometLogger
from torch import FloatTensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import NumpyVoxelDataset, ShapeNetVoxelDataset
from utils import get_free_gpu, searcher, validate_conv


class DenoisingAutoEncoder(pl.LightningModule):
    logger: CometLogger

    def __init__(self, hparams: dict, train_set: Tuple[str, str], val_set: Tuple[str, str]):
        super().__init__()

        self.hparams = Namespace(**hparams)
        self.vector_length = self.hparams.vector_length
        self.dim = self.hparams.dim
        self.lr = self.hparams.lr
        self.bs = self.hparams.batch_size
        self.normal_mean = self.hparams.normal_mean
        self.normal_sigma = self.hparams.normal_sigma
        self.conv_layers = self.hparams.conv_layers
        self.bn = self.hparams.bn
        self.elu = self.hparams.elu
        self.kld_ratio = self.hparams.kld_ratio
        self.weighted_loss = self.hparams.weighted_loss

        self.conv_size = validate_conv(self.dim, self.conv_layers)
        self.conv_channel = self.conv_layers[-1][1]
        if not self.conv_size:
            raise RuntimeError("Conv layers not aligned.")

        self.encoder = nn.Sequential(
            # Example:
            # conv_layers=((1, 10, 2, 2, 0), (10, 20, 2, 2, 0))
            # nn.Sequential(
            #   nn.Conv3d(1, 10, 2, 2, 0),
            #   nn.LeakyRelu(),
            #   nn.Conv3d(10, 20, 2, 2, 0),
            #   nn.LeakyRelu()
            # )
            *(layer for layers in (
                [nn.Conv3d(*conv_layer)] +
                ([nn.BatchNorm3d(conv_layer[1])] if self.bn else []) +
                [nn.ELU() if self.elu else nn.ReLU()]
                for conv_layer in self.conv_layers)
              for layer in layers)
        )

        self.mu = nn.Sequential(
            nn.Linear(pow(self.conv_size, 3) * self.conv_channel, self.vector_length),
        )
        self.log_var = nn.Sequential(
            nn.Linear(pow(self.conv_size, 3) * self.conv_channel, self.vector_length),
        )

        self.fc_decoder = nn.Sequential(
            nn.Linear(self.vector_length, pow(self.conv_size, 3) * self.conv_channel),
            nn.ELU() if self.elu else nn.ReLU()
        )

        self.decoder = nn.Sequential(
            *(layer for layers in (
                [nn.ConvTranspose3d(conv_layer[1], conv_layer[0], *conv_layer[2:])] +
                ([nn.BatchNorm3d(conv_layer[0])] if self.bn else []) +
                [nn.ELU() if self.elu else nn.ReLU()]
                for conv_layer in self.conv_layers[:0:-1])
              for layer in layers),
            nn.ConvTranspose3d(self.conv_layers[0][1], self.conv_layers[0][0], *self.conv_layers[0][2:]),
            nn.Sigmoid()
        )

        # Data
        # self.train_dataset = NumpyVoxelDataset(train_set[0], self.dim, train_set[1])
        # self.val_dataset = NumpyVoxelDataset(val_set[0], self.dim, val_set[1])
        # self.val_dataset = PickerDataset(self.train_dataset, ids=[10, 513, 2013, 10403, 20303, 40013])
        self.train_dataset = ShapeNetVoxelDataset('shapenetvoxel128', 'pix2mesh_splits_val05.json', 128, 'train')
        self.val_dataset = ShapeNetVoxelDataset('shapenetvoxel128', 'pix2mesh_splits_val05.json', 128, 'test')

    @staticmethod
    def reparameterize(mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_encoder(self, x):
        x = self.encoder(x)
        x = x.flatten(1)
        mu, log_var = self.mu(x), self.log_var(x)
        return mu, log_var

    def forward_decoder(self, x):
        x = self.fc_decoder(x)
        x = x.view(-1, self.conv_channel, *([self.conv_size] * 3))
        recon = self.decoder(x)
        return recon

    def forward(self, x):
        mu, log_var = self.forward_encoder(x)
        hidden = self.reparameterize(mu, log_var)
        recon = self.forward_decoder(hidden)
        return recon, mu, log_var

    def prepare_data(self):
        self.logger.experiment.set_model_graph("\n".join((str(layer) for layer in (
            self.encoder,
            self.mu,
            self.log_var,
            self.fc_decoder,
            self.decoder
        ))))

    def loss(self, recon, x, mu, log_var):
        bce = F.binary_cross_entropy(recon, x, reduction='sum', weight=x + 0.1 if self.weighted_loss else None)
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        mean = (bce + self.kld_ratio * kld) / reduce(operator.mul, x.shape)
        return mean

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.bs,
                          shuffle=True,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.bs,
                          shuffle=False,
                          pin_memory=True)

    def configure_optimizers(self):
        # return torch.optim.SGD(self.parameters(),
        #                        lr=0.001, momentum=0.99, weight_decay=0.0005)
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _step(self, image: FloatTensor, noise: bool = True) \
            -> Tuple[FloatTensor, FloatTensor, FloatTensor, FloatTensor]:
        if noise:
            # noinspection PyArgumentList
            gauss = FloatTensor(*image.shape).to(image.device).normal_(mean=self.normal_mean, std=self.normal_sigma)
            noised = image + gauss
        else:
            noised = image

        # TODO: Check Shape
        recon, mu, log_var = self(noised)

        return image, recon, mu, log_var

    def training_step(self, batch, batch_idx):
        image, recon, mu, log_var = self._step(batch)

        loss = self.loss(recon, image, mu, log_var)

        _max = recon.max()
        logs = {'loss': loss, 'max': _max}
        return {**logs, "log": logs}

    def validation_step(self, batch, batch_idx):
        image, recon, mu, log_var = self._step(batch, noise=False)

        loss = self.loss(recon, image, mu, log_var)

        logs = {"val_loss": loss}
        return logs

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        logs = {'val_loss': val_loss_mean}
        return {**logs, "log": logs}


if __name__ == '__main__':
    CUDA_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
    if CUDA_DEVICES:
        gpus = [CUDA_DEVICES.split(",")]
    else:
        gpus = [get_free_gpu()]

    with open("tokens.json", mode="r") as f:
        conf = json.load(f)

    # search structure
    # in_channels, out_channels, kernel_size, stride, padding
    conv_layers_grid = [
        [
            (1, 32, 8, 4, 2),
            (32, 64, 4, 2, 1),
            (64, 128, 4, 2, 1),
            (128, 128, 4, 2, 1)
        ],
        # [
        #     (1, 15, 4, 2, 1),
        #     (15, 30, 4, 2, 1),
        #     (30, 50, 4, 2, 1),
        #     (50, 100, 4, 2, 1),
        #     (100, 100, 4, 2, 1),
        # ],
    ]

    # validate conv layers
    for conv_layers in conv_layers_grid:
        dim = validate_conv(128, conv_layers)
        if not dim:
            raise RuntimeError("Conv layers not aligned.")

    # search hparams
    hparams_grid = {
        "dim": 128,
        "vector_length": 128,
        "normal_mean": 0,
        "normal_sigma": 0.05,
        "lr": 1e-4,
        "batch_size": 64,
        "conv_layers": conv_layers_grid,
        "bn": False,
        "elu": True,
        "kld_ratio": [0.1, 1],
        "weighted_loss": [True, False]
    }

    search_params = list(searcher(hparams_grid))
    for idx, hparams in enumerate(search_params):
        print(f"Progress: {idx}/{len(search_params)}")
        print("Searching hparams: ", hparams)
        comet_logger = CometLogger(**conf["comet"])
        model = DenoisingAutoEncoder(**conf["data"], hparams=hparams)
        # model = DenoisingAutoEncoder(hparams=hparams, data_path='voxel.npz', array_name="arr_0")

        ckpt_cb = ModelCheckpoint(
            filepath="".join(["ckpt/", datetime.datetime.now().isoformat().replace(':', '.'), "/{epoch}"]),
            save_top_k=-1,
            period=5
        )

        print(f"Selecting gpus:{gpus}")
        trainer = Trainer(gpus=gpus, logger=comet_logger, max_epochs=80, checkpoint_callback=ckpt_cb,
                          num_sanity_val_steps=1, check_val_every_n_epoch=5)
        trainer.fit(model)
