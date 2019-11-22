import os
import numpy as np
from PIL import Image
import torch
from scipy import ndimage


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
        mask_array = np.array(mask)
        # make the picture binary. 100 is arbitrary but seems to work for dataset
        mask_array[np.where(mask_array > 100)] = 255
        mask_array[np.where(mask_array <= 100)] = 0

        boxes = []
        # find different objects (roofs) in mask
        label_img, nr_labels = ndimage.label(mask_array)
        for i in range(1, nr_labels + 1):
            pos = np.where(label_img == i)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        labels = torch.ones((nr_labels,), dtype=torch.int64)

        # instances are encoded as different colors
        obj_ids = np.unique(label_img)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = label_img == obj_ids[:, None, None]

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((nr_labels,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd (artifact from coco pretrained net)
        iscrowd = torch.zeros((nr_labels,), dtype=torch.int64)

        # suppose all instances are not crowd

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def save_color_encoded_images(self, subfolder='color_encoded'):
        color_codes = [10, 40, 70, 100, 130, 160, 190, 220, 250]

        subfolder_path = os.path.join(self.root, subfolder)
        if not os.path.exists(subfolder_path):
            os.mkdir(subfolder_path)
        for mask_file_name in self.masks:
            print(f'filename: {mask_file_name}')
            mask_path = os.path.join(self.root, "labels", mask_file_name)
            with Image.open(mask_path) as mask:
                mask = mask.convert('L')
            mask_array = np.array(mask)
            # make the picture binary. 100 is arbitrary but seems to work for dataset
            mask_array[np.where(mask_array > 100)] = 255
            mask_array[np.where(mask_array <= 100)] = 0
            label_img, nr_labels = ndimage.label(mask_array)

            colored = mask_array
            print(f'nr labels: {nr_labels}')
            for i in range(1, nr_labels + 1):
                colored[np.where(label_img == i)] = color_codes[i-1]
            img = Image.fromarray(colored, 'L')
            img.save(os.path.join(subfolder_path, mask_file_name))

    def __len__(self):
        return len(self.imgs)


# data_root_path = 'images'
# dataset = RooftopDataset(root=data_root_path, grayscale=True, transforms=None)
# img, mask = dataset[0]