import numpy as np
from PIL import Image
import torch


class Compose(object):
    def __init__(self, np_aug=None, pil_aug=None, to_pil=False):
        self.pil_augmentations = pil_aug
        self.np_augmentations = np_aug
        self.pil = True
        self.to_pil = to_pil

    def __call__(self, img):
        # first do augmentations implemented by PIL
        if self.pil_augmentations is not None:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img.astype(np.uint8))
                self.pil = True
            for a in self.pil_augmentations:
                img = a(img)
        if self.np_augmentations is not None:
            if not isinstance(img, np.ndarray):
                img = np.array(img, dtype=np.uint8)
                self.pil = False
            for a in self.np_augmentations:
                img = a(img)

        # convert back to PIL
        if self.to_pil and not self.pil:
            img = Image.fromarray(img)

        return img


class AddRandomNoise(object):
    def __init__(self, sigma=0.05):
        self.mean = 0
        self.sigma = sigma

    def __call__(self, img):
        gauss = np.random.normal(self.mean, self.sigma, img.shape) * (self.sigma * 2)
        return (img + gauss).clip(0, 1)


class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return torch.tensor(x).float()
