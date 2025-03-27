
from data_preprocessing_detr import get_data
from visualisations import visualisations

from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models.detr import Detr
from training.train_eval import *

# Set paths and parameters
DATASET_DIR = os.getenv("DATASET_DIR", "data")  
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")  
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# Get DataLoaders and test DataLoader
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

#Define model
model = Detr()
sys.stdout.flush()

# Train
# model.train()
# model = train_model_pretrained(dls.train)
# visualisations.visualize_batch(dls, OUTPUT_DIR, 50, 10)

# Save Model
# print("STARTING")
# detr_model_bbox_pretrained_10_correct was (height, width).T
# detr_model_bbox_pretrained_10_correct_better was (width, height).T, which seems better (10 epochs)
# detr_model_bbox_pretrained_10_correct_better_epoch1.pth same as bove but 1 epoch
# detr_model_bbox_pretrained_10_correct_better_epoch5.pth same as above but 5 epochs (Smaller amounts of data)
# detr_model_bbox_pretrained_10_correct_better_epoch5_b.pth same as above but 5 epochs with all data
model_path = os.path.join(OUTPUT_DIR, "detr_model_bbox_pretrained_10_correct_better_epoch1.pth")
# torch.save(model.state_dict(), model_path)
# print("Model saved")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"), weights_only=True), strict=False)

# Move the model to the appropriate device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Ensure the model is in evaluation mode
model.eval()

#Test
acc_test = test_model(model, dls.train, 1000, OUTPUT_DIR)

