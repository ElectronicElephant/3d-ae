import numpy as np
from PIL import Image


class Compose(object):
    def __init__(self, np_aug=None, pil_aug=None):
        self.pil_augmentations = pil_aug
        self.np_augmentations = np_aug
        self.pil = True

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
        if not self.pil:
            img = Image.fromarray(img)

        return img


class AddRandomNoise(object):
    def __init__(self, sigma=0.05):
        self.mean = 0
        self.sigma = sigma

    def __call__(self, img):
        gauss = np.random.normal(self.mean, self.sigma, img.shape) * (255 * self.sigma * 2)
        return (img + gauss).astype(np.uint8)
