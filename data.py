import json
import os
from functools import lru_cache
from typing import List

import numpy as np
import torch
from torch.utils import data
from tqdm import tqdm


# TODO: @LightQuantum: Rewrite the whole bunch of dataloader in shapenet format
class NumpyVoxelDataset(data.Dataset):
    def __init__(self, data_path, dim, array_name=None):
        self.data_path = data_path
        self.array_name = array_name
        self.dim = dim if isinstance(dim, tuple) else (dim, dim, dim)

        # Dataset
        if not os.path.exists(self.data_path):  # TODO: Move to config!
            raise FileNotFoundError('File %s does not exist!' % self.data_path)

        self.samples = np.load(data_path)
        if array_name:
            self.samples = self.samples[array_name]

        if len(self.samples.shape) != 4:
            raise RuntimeError('The shape of the array is %s which is not supported.'
                               % self.samples.shape)
        if not (self.samples.shape[-1] == self.samples.shape[-2] == self.samples.shape[-3] == dim):
            raise RuntimeError('The shape of the array is %s which is not equal to %s.'
                               % (self.samples.shape, self.dim))

    def __len__(self):
        return len(self.samples)

    @lru_cache(None)
    def __getitem__(self, index) -> torch.FloatTensor:
        # noinspection PyArgumentList
        return torch.FloatTensor([self.samples[index].astype(np.float)])  # C, D, H, W


class PickerDataset(data.Dataset):
    def __init__(self, base_set: data.Dataset, ids: List[int]):
        self.base_set = base_set
        if max(ids) >= len(self.base_set):
            raise ValueError("Index outbound.")
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    @lru_cache(maxsize=None)
    def __getitem__(self, item):
        return self.base_set[self.ids[item]]


class DAENumpyVoxelDataset(NumpyVoxelDataset):
    def __init__(self, data_path, dim, array_name=None, test_mode=False,
                 mean=0, sigma=0.05, clip=True, low=0, high=1):
        super().__init__(data_path, dim, array_name=array_name)
        self.test_mode = test_mode
        self.mean = mean
        self.sigma = sigma
        self.clip = clip
        self.low = low
        self.high = high

    def __getitem__(self, index):
        if self.test_mode:
            # return only the origin image
            return self.to_tensor([self.samples[index].astype(np.float)])  # C, D, H, W
        else:
            image = self.samples[index].astype(np.float)
            return (self.to_tensor([image]),
                    self.to_tensor([self.add_noise(image)]))

    @staticmethod
    def to_tensor(x):
        """
        torch.transforms.ToTensor() Does NOT support 3D-Image
        This function support arbitrary-shaped tensor
        """
        return torch.tensor(x).float()

    def add_noise(self, img):
        gauss = np.random.normal(self.mean, self.sigma, img.shape)
        if self.clip:
            return (img + gauss).clip(self.low, self.high)
        return img + gauss


class ShapeNetVoxelDataset(data.Dataset):
    def __init__(self, data_path, json_name, dim, array_name=None):
        self.data_path = data_path
        self.array_name = array_name
        self.dim = dim if isinstance(dim, tuple) else (dim, dim, dim)

        # Dataset
        if not os.path.exists(self.data_path):  # TODO: Move to config!
            raise FileNotFoundError('File %s does not exist!' % self.data_path)

        self.data = None
        self.json_name = json_name
        with open(os.path.join(self.data_path, self.json_name)) as f:
            self.data = json.load(f)

        assert self.data is not None

        self.annos = {
            k: list(self.data[self.array_name][k].keys())
            for k in self.data[self.array_name]
        }

        self.samples = []
        for (cat_id, item_ids) in self.annos.items():
            for item_id in tqdm(item_ids):
                dense = np.load(os.path.join(self.data_path, cat_id, item_id, 'voxel.npz'))['filled_dense']
                if dense.shape[0] == self.dim[0] and dense.shape[1] == self.dim[1] and dense.shape[2] == self.dim[2]:
                    pass
                else:
                    raise RuntimeError('Data shape not unified!')
                self.samples.append(dense)

    def __len__(self):
        return len(self.samples)

    @lru_cache(None)
    def __getitem__(self, index) -> torch.FloatTensor:
        # noinspection PyArgumentList
        return torch.FloatTensor([self.samples[index].astype(np.float)])  # C, D, H, W
