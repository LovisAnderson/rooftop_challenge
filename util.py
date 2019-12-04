import numpy as np
from matplotlib import pyplot as plt


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    n = len(images)
    plt.figure(figsize=(16, 5))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image)
    plt.show()


def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def save_predictions(dataset, model, out_path='images/results'):
    for i in range(len(dataset)):
        image, gt_mask = dataset[i]
        image_name = dataset.ids[i]
        pr_mask = model.predict(image).round()
        plt.imsave(f'{out_path}/{image_name}', pr_mask[..., 0].squeeze())