import albumentations as A

# ### Augmentations

# Data augmentation is a powerful technique to increase the amount of your data and prevent model overfitting.
# If you not familiar with such trick read some of these articles:
#  - [The Effectiveness of Data Augmentation in Image Classification using Deep
# Learning](http://cs231n.stanford.edu/reports/2017/pdfs/300.pdf)
#  - [Data Augmentation | How to use Deep Learning when you have Limited Data](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced)
#  - [Data Augmentation Experimentation](https://towardsdatascience.com/data-augmentation-experimentation-3e274504f04b)
#
# Since our dataset is very small we will apply a large number of different augmentations:
#  - horizontal flip
#  - affine transforms
#  - perspective transforms
#  - brightness/contrast/colors manipulations
#  - image bluring and sharpening
#  - gaussian noise
#  - random crops
#
# All this transforms can be easily applied with [**Albumentations**](https://github.com/albu/albumentations/) - fast augmentation library.
# For detailed explanation of image transformations you can look at [kaggle salt segmentation exmaple](https://github.com/albu/albumentations/blob/master/notebooks/example_kaggle_salt.ipynb) provided by [**Albumentations**](https://github.com/albu/albumentations/) authors.
#


# define heavy augmentations


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


def get_training_augmentation():
    train_transform = [

        A.HorizontalFlip(p=0.5),

        A.ShiftScaleRotate(scale_limit=0.0, rotate_limit=360, shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),
        A.RandomCrop(height=256, width=256, always_apply=True),

        # A.IAAAdditiveGaussianNoise(p=0.2),
        # Satellite pictures always have the same perspective,
        # Therefore we do not want our model to learn the concept of perspective
        # A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


# christchurch dataset is large and we do not need to augment it that much.
# Since the pictures are taken from farer away and a larger we want to crop first
# For better training we crop images where there is a non-empty mask


def get_cc_training_augmentation():
    train_transform = [
        A.CropNonEmptyMaskIfExists(height=640, width=640, always_apply=True),
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.PadIfNeeded(min_height=256, min_width=256, always_apply=True, border_mode=0),

    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(256, 256)
    ]
    return A.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)