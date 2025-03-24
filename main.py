
from data_preprocessing_detr import get_data
from visualisations import visualisations

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
bs = 32  # batch size
nw = 4    # number of workers for data loader

# Get DataLoaders and test DataLoader using the new get_data function
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

#Define model
model = Detr()
# print("Starting training...")  # Check if training actually starts
# sys.stdout.flush()
# model.train()

# Train
# train_model(model, dls.train)
# visualisations.visualize_batch(dls, OUTPUT_DIR, 30, 10)

#Save Model
# Last one was bbox
print("STARTING")
model_path = os.path.join(OUTPUT_DIR, "detr_model_bbox_thresh.pth")
# torch.save(model.state_dict(), model_path)
# print(f"Model saved to {model_path}")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True), strict=False)

# Move the model to the appropriate device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

#Test

acc_test = test_model(model, dls.train, 30, OUTPUT_DIR)

