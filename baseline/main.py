
from data_preprocessing import get_data

# Fast.ai v2 unified imports
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Set paths and parameters
PATH = './'
TRAIN = 'data/train_v2'
TEST = 'data/test_v2'
SEGMENTATION = 'data/train_ship_segmentations_v2.csv'
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']  # corrupted images

sz = 256  # image size
bs = 64   # batch size
nw = 4    # number of workers for data loader

arch = resnet34

# Get DataLoaders and test DataLoader using the new get_data function
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

# Create a learner.

# The parameter ps=0.5 will set the dropout in the head to 50%.
learn = cnn_learner(dls, arch, metrics=accuracy, ps=0.5)

# Set the optimizer to Adam (this is usually the default, but you can specify it explicitly)
learn.opt_func = Adam

# Optionally: print a summary of the model
learn.model.eval()
learn.model

# Train the model – for example, fine-tune for 3 epochs:
learn.fine_tune(3)

# Optionally, obtain predictions on the test set using Test Time Augmentation (TTA)
preds, targs = learn.tta(dl=test_dl)
print(preds)