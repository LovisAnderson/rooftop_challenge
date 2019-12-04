#!/usr/bin/env python
# coding: utf-8

# ### Reqirements
# - keras >= 2.2.0 or tensorflow >= 1.13
# - segmenation-models==1.0.*
# - albumentations==0.3.0



import os

from .augmentation import get_training_augmentation, get_cc_training_augmentation, get_preprocessing
from .data import Dataset, Dataloder
from .util import save_predictions


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import keras
import segmentation_models as sm


DATA_DIR = 'images/'

CHRISTCHURCH_DIR = 'trainval'

x_cc_train_dir = os.path.join(CHRISTCHURCH_DIR, 'train/image')
y_cc_train_dir = os.path.join(CHRISTCHURCH_DIR, 'train/label')

x_dida_train_dir = os.path.join(DATA_DIR, 'frames/train')
y_dida_train_dir = os.path.join(DATA_DIR, 'masks/train')

x_valid_dir = os.path.join(DATA_DIR, 'validation/frames')
y_valid_dir = os.path.join(DATA_DIR, 'validation/masks')


# # Segmentation model training

# segmentation_models could also use `tf.keras` if you do not have Keras installed
# or you could switch to other framework using `sm.set_framework('tf.keras')`

BACKBONE = 'efficientnetb3'
BATCH_SIZE = 2
BATCH_SIZE_CC = 16
CLASSES = ['rooftop']
LR = 0.0005
EPOCHS = 20
EPOCHS_CC = 20

preprocess_input = sm.get_preprocessing(BACKBONE)


# define network parameters
n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
activation = 'sigmoid' if n_classes == 1 else 'softmax'

# create model
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

# load existing weights
model.load_weights('best_model.h5')

# define optimizer
optim = keras.optimizers.Adam(LR)

# Segmentation models losses can be combined together by '+' and scaled by integer or float factor


# Using BinaryFocalLoss
total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss
total_loss = sm.losses.BinaryFocalLoss()
metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

# compile keras model with defined optimozer, loss and metrics
model.compile(optim, total_loss, metrics)


# Dataset for train images on dida data
print('Model compiled!')


# Dataset for train images
train_dataset = Dataset(
    x_dida_train_dir,
    y_dida_train_dir,
    classes=['rooftop'],
    preprocessing=get_preprocessing(preprocess_input),
    augmentation=get_training_augmentation()
)

# Dataset for train images on christchurch data
train_dataset_cc = Dataset(
    x_cc_train_dir,
    y_cc_train_dir,
    classes=['rooftop'],
    preprocessing=get_preprocessing(preprocess_input),
    augmentation=get_cc_training_augmentation()
)

# Dataset for validataion on dida data
valid_dataset = Dataset(
    x_valid_dir,
    y_valid_dir,
    classes=CLASSES,
    # augmentation=get_validation_augmentation(),
    preprocessing=get_preprocessing(preprocess_input),
)


train_dataloader = Dataloder(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
train_dataloader_cc = Dataloder(train_dataset_cc, batch_size=BATCH_SIZE_CC, shuffle=True)

valid_dataloader = Dataloder(valid_dataset, batch_size=1, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (BATCH_SIZE, 256, 256, 3)
assert train_dataloader[0][1].shape == (BATCH_SIZE, 256, 256, n_classes)
assert train_dataloader_cc[0][0].shape == (BATCH_SIZE_CC, 256, 256, 3)
assert train_dataloader_cc[0][1].shape == (BATCH_SIZE_CC, 256, 256, n_classes)


# define callbacks for learning rate scheduling and best checkpoints saving
callbacks = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.TensorBoard(update_freq='epoch')
]

callbacks_cc = [
    keras.callbacks.ModelCheckpoint('./best_model.h5', save_weights_only=True, save_best_only=True, mode='min'),
    keras.callbacks.ReduceLROnPlateau(),
    keras.callbacks.TensorBoard(update_freq=100)
]

# train model on christchurch data
history_cc = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader_cc),
    epochs=EPOCHS_CC,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)

# train model on dida data to fit better
history = model.fit_generator(
    train_dataloader,
    steps_per_epoch=len(train_dataloader),
    epochs=EPOCHS,
    callbacks=callbacks,
    validation_data=valid_dataloader,
    validation_steps=len(valid_dataloader),
)


# Create and save result images
model.load_weights('best_model.h5')
save_predictions(valid_dataset, model)



# # Visualization of results on test dataset

# n = 5
# ids = np.random.choice(np.arange(len(test_dataset)), size=n)
#
# for i in ids:
#
#     image, gt_mask = test_dataset[i]
#     image = np.expand_dims(image, axis=0)
#     pr_mask = model.predict(image).round()
#
#     visualize(
#         image=denormalize(image.squeeze()),
#         gt_mask=gt_mask[..., 0].squeeze(),
#         pr_mask=pr_mask[..., 0].squeeze(),
#     )
#
