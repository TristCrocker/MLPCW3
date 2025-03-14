
from data_preprocessing_detr import get_data

# Fast.ai v2 unified imports
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models.detr import Detr
from training.train_eval import *

# Set paths and parameters
DATASET_DIR = os.getenv("DATASET_DIR", "data")  # Default to "data" if not set
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")  # Default to "output" if not set
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

PATH = 'output'
TRAIN = os.path.join("data", "train_v2")
TEST = os.path.join("data", "test_v2")
SEGMENTATION = os.path.join("data/segmentations", "train_ship_segmentations_v2.csv")
exclude_list = []  # corrupted images

sz = 256  # image size
bs = 1   # batch size
nw = 4    # number of workers for data loader


# Get DataLoaders and test DataLoader using the new get_data function
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

#Define model
model = Detr(num_queries=100)
model.train()

#Train
train_model(model, dls.train)

#Save Model
model_path = os.path.join("output", "detr_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

#Test
acc_test = test_model(model, test_dl)

