import os
import numpy as np
from PIL import Image
import torch
from scipy import ndimage
from torchvision.transforms import functional as F
import random


class RooftopDataset(object):
    def __init__(self, root, grayscale=False, transforms=None, label_subfolder="color_encoded"):
        self.root = root
        self.grayscale = grayscale
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, label_subfolder))))
        self.label_subfolder = label_subfolder

    def __getitem__(self, idx):
        # load images and masks

        img_path = os.path.join(self.root, "images", self.imgs[idx])
        mask_path = os.path.join(self.root, self.label_subfolder, self.masks[idx])
        with Image.open(img_path) as img:
            if self.grayscale:
                img = img.convert("L")
            else:
                img = img.convert("RGB")

        mask = Image.open(mask_path)
        if self.transforms is not None:
            # In order to get the same transformations for img and target we use the same seed for both
            RANDOM_SEED = np.random.randint(2147483647)
            random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            img = self.transforms(img)
            random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            mask = self.transforms(mask)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd (artifact from coco pretrained net)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        img = F.to_tensor(img)

        return img, target

    def save_color_encoded_images(self, subfolder='color_encoded', treshold=150):

        subfolder_path = os.path.join(self.root, subfolder)
        if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
        for mask_file_name in self.masks:
            print(f'filename: {mask_file_name}')
            mask_path = os.path.join(self.root, "labels", mask_file_name)
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
            mask_array = np.array(mask)
            # make the picture binary.  is arbitrary but seems to work for dataset
            # There are grey pixel at the boundary we use treshold to decide if they are white or black
            # treshold=150 correctly identified all rooftops
            # todo: borders might be identified as non-roof even if they belong to the roof. Investigate!
            mask_array[np.where(mask_array > treshold)] = 255
            mask_array[np.where(mask_array <= treshold)] = 0
            label_img, nr_labels = ndimage.label(mask_array)

            colored = mask_array
            print(f'nr labels: {nr_labels}')
            for i in range(1, nr_labels + 1):
                colored[np.where(label_img == i)] = i
            img = Image.fromarray(colored)
            img.save(os.path.join(subfolder_path, mask_file_name))

    def __len__(self):
        return len(self.imgs)


# data_root_path = 'images'
# dataset = RooftopDataset(root=data_root_path, grayscale=True, transforms=None)
# img, mask = dataset[0]