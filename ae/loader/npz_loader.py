from torchvision import transforms

from torch.utils import data
import numpy as np


# TODO: @LightQuantum: Rewrite the whole bunch of dataloader in shapenet format
class NPZLoader(data.Dataset):
    def __init__(self, npz_path, array_name, ae_type="dae", is_transform=False, dim=32, augmentations=None,
                 test_mode=False):
        self.npz_path = npz_path
        self.array_name = array_name
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.dim = dim if isinstance(dim, tuple) else (dim, dim, dim)
        self.ae_type = (ae_type == "dae")
        self.n_classes = 1
        self.test_mode = test_mode

        self.samples = np.load(npz_path)
        if array_name:
            self.samples = self.samples[array_name]
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.test_mode:
            sample = self.transform(sample)
            # (image, feature) pair
            return [sample, '']

        if self.ae_type:
            sample, noisy_sample = self.transform(sample)
            return [sample, noisy_sample]
        else:
            sample = self.transform(sample)
            return [sample]

    def transform(self, image):
        if self.test_mode:
            return self.tf(image)

        if self.ae_type:
            noisy_image = self.augmentations(image)
            noisy_image = self.tf(noisy_image)
        image = self.tf(image)
        if self.ae_type:
            return image, noisy_image
        else:
            return image
