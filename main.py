
from data_preprocessing_detr import get_data

# Fast.ai v2 unified imports
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models.detr import Detr
from training.train_eval import *

if torch.cuda.is_available():
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("No GPU detected. Running on CPU!")

# Set paths and parameters
DATASET_DIR = os.getenv("DATASET_DIR", "data")  # Default to "data" if not set
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")  # Default to "output" if not set
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure output directory exists

PATH = 'output'
TRAIN = os.path.join(DATASET_DIR, "train_v2")
TEST = os.path.join(DATASET_DIR, "test_v2")
SEGMENTATION = os.path.join(DATASET_DIR, "train_ship_segmentations_v2.csv")
exclude_list = ['6384c3e78.jpg','13703f040.jpg', '14715c06d.jpg', '33e0ff2d5.jpg',
                '4d4e09f2a.jpg', '877691df8.jpg', '8b909bb20.jpg', 'a8d99130e.jpg', 
                'ad55c3143.jpg', 'c8260c541.jpg', 'd6c7f17c7.jpg', 'dc3e7c901.jpg',
                'e44dffe88.jpg', 'ef87bad36.jpg', 'f083256d8.jpg']  # corrupted images

sz = 256  # image size
bs = 64  # batch size
nw = 4    # number of workers for data loader


# Get DataLoaders and test DataLoader using the new get_data function
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

#Define model
model = Detr(num_queries=100)
model.train()

#Train
train_model(model, dls.train)

#Save Model
model_path = os.path.join(OUTPUT_DIR, "detr_model.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

#Test
acc_test = test_model(model, test_dl)

