
from data_preprocessing_detr import get_data

# Fast.ai v2 unified imports
from fastai.vision.all import *
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

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
bs = 64   # batch size
nw = 4    # number of workers for data loader

arch = resnet34

# Get DataLoaders and test DataLoader using the new get_data function
dls, test_dl = get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list)

# Create a learner.

# The parameter ps=0.5 will set the dropout in the head to 50%.
learn = vision_learner(dls, arch, metrics=accuracy, ps=0.5)

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

# Evaluate the model on the validation set
eval_results = learn.validate()

print(eval_results)
# Extract metric names from the learner
metric_names = ["Loss"] + [m.name if hasattr(m, "name") else str(m) for m in learn.metrics]

# Create a DataFrame for evaluation results
df_eval = pd.DataFrame([eval_results], columns=metric_names)

# Define output file path (ensure it's an accessible location)
output_csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")

# Save DataFrame to a CSV file
df_eval.to_csv(output_csv_path, index=False)

print(f"Model evaluation results saved to {output_csv_path}")