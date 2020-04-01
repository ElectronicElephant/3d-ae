import collections
from os.path import join as pjoin

from PIL import Image
from torch.utils import data
from torchvision import transforms


class BasicLoader(data.Dataset):
    def __init__(self, root, train_file, ae_type="dae", is_transform=False, img_size=64, augmentations=None,
                 test_mode=False):
        self.root = root
        self.is_transform = is_transform
        self.train_file = train_file
        self.files = collections.defaultdict(list)
        self.augmentations = augmentations
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.ae_type = (ae_type == "dae")
        self.n_classes = 1
        self.test_mode = test_mode

        path = pjoin(self.root, train_file)
        file_list = tuple(open(path, "r"))
        self.files = [id_.rstrip() for id_ in file_list]
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img_name = self.files[index]
        img_path = pjoin(self.root, img_name)
        image = Image.open(img_path)

        if self.test_mode:
            image = self.transform(image)
            # (image, feature) pair
            return [image, img_name.split('/')[1]]

        if self.ae_type:
            image, noisy_image = self.transform(image)
            return [image, noisy_image]
        else:
            image = self.transform(image)
            return [image]

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
