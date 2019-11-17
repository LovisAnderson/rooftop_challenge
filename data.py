import os
import numpy as np
import torch
from PIL import Image


class RooftopDataset(object):
    def __init__(self, root, grayscale=False, transforms=None):
        self.root = root
        self.grayscale = grayscale
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        # load images ad masks

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, "labels", self.masks[idx])
        with Image.open(img_path) as img:
            if self.grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")
        with Image.open(mask_path) as mask:
            mask = mask.convert('L')
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # convert the PIL Images into  numpy arrays

        img = np.array(img)
        mask = np.array(mask)
        mask = mask.reshape((1, 256, 256))
        if self.grayscale:
            img = img.reshape(1, 256, 256)

        # Non-zero means that this pixel belongs to a rooftop
        mask[np.where(mask != 0)] = 1

        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)