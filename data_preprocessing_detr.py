import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from fastai.vision.all import *
from torchvision.transforms.functional import resize
import torchvision

def train_set_split(PATH, TRAIN, TEST, SEGMENTATION, exclude_list):
    """
    Splits the dataset into train/validation (5% validation) and removes excluded files.
    """
    train_names = [f for f in os.listdir(TRAIN)]
    global test_names
    test_names = [f for f in os.listdir(TEST)]
    
    # Remove excluded files
    train_names = [f for f in train_names if f not in exclude_list]
    test_names = [f for f in test_names if f not in exclude_list]

    # 5% of data in validation set
    tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)

    return tr_n, val_n

def rle_to_mask(rle, height=768, width=768):
    """
    Convert run-length encoding (RLE) to a binary mask.
    """
    mask = np.zeros(height * width, dtype=np.uint8)
    if isinstance(rle, float) or pd.isna(rle):
        return mask.reshape((width, height))

    rle = list(map(int, rle.split()))
    starts, lengths = rle[0::2], rle[1::2]

    for start, length in zip(starts, lengths):
        mask[start: start + length] = 1

    return mask.reshape((width, height)).T

def mask_to_bbox(mask, height=256, width=256):
    y_indices, x_indices = np.where(mask == 1)

    if len(y_indices) == 0 or len(x_indices) == 0:
        return None  # No ship found

    # Convert mask pixels to bounding box corners
    xmin, xmax = x_indices.min(), x_indices.max() + 1
    ymin, ymax = y_indices.min(), y_indices.max() + 1

    w = xmax - xmin + 1
    h = ymax - ymin + 1
    cx = (xmin + xmax) / 2  # Center X
    cy = (ymin + ymax) / 2  # Center Y

    return torch.tensor([cx / width, cy / height, w / width, h / height], dtype=torch.float32) #Detr format output for bbox


def label_func(fname, seg_df, height=256, width=256, num_queries=20, min_size=10):

    if fname not in seg_df.index:
        return torch.zeros((num_queries, 4)), torch.zeros(num_queries, dtype=torch.long)

    masks = seg_df.loc[fname, 'EncodedPixels']

    # Ensure masks is a list or Series, not a bool
    if isinstance(masks, pd.Series):
        masks = masks.dropna().tolist()  # Convert to list & remove NaNs
    elif isinstance(masks, float) or pd.isna(masks):
        masks = []  # Convert NaN to an empty list
    elif isinstance(masks, str):  
        masks = [masks]  # Convert a single mask string to a list
    
    # Check if there are no valid masks
    if len(masks) == 0:
        return torch.zeros((num_queries, 4)), torch.zeros(num_queries, dtype=torch.long)

    # Create bboxes, where we find mask from 768x768 image, rescale mask to 256x256 and then find bbox in terms of 256x256 iamge
    bboxes = []
    for mask in masks:
        full_mask = rle_to_mask(mask, height=768, width=768) 
        
        full_mask_tensor = torch.tensor(full_mask).unsqueeze(0).float() 

        resized_mask = resize(full_mask_tensor, size=(height, width), interpolation=torchvision.transforms.InterpolationMode.NEAREST).squeeze(0).byte() 

        bbox = mask_to_bbox(resized_mask.numpy(), height, width)

        if bbox is not None:
            if bbox[2] > min_size / width and bbox[3] > min_size / height:
                bboxes.append(bbox)

    # If no valid bounding boxes were found
    if len(bboxes) == 0:
        return torch.zeros((num_queries, 4)), torch.zeros(num_queries, dtype=torch.long)

    bboxes = torch.stack(bboxes) if len(bboxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
    class_labels = torch.ones(len(bboxes), dtype=torch.long)  # 1 = Ship class

    # Pad bounding boxes & labels to num_queries
    padded_bboxes = torch.zeros((num_queries, 4), dtype=torch.float32)
    padded_labels = torch.zeros(num_queries, dtype=torch.long)

    num_valid = min(len(bboxes), num_queries)
    padded_bboxes[:num_valid] = bboxes[:num_valid]
    padded_labels[:num_valid] = class_labels[:num_valid]

    return padded_bboxes, padded_labels


def get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list, num_queries=10):
    """

    Arguments:
    sz (int) : image size
    bs (int) : batch size
    PATH (str) : root path
    TRAIN (str): path to training images
    TEST (str) : path to test images
    SEGMENTATION (str): CSV with 'ImageId' and 'EncodedPixels'
    exclude_list (list): filenames to exclude

    Returns:
    dls (DataLoaders): train/valid DataLoaders
    test_dl (TfmdDL) : optional test DataLoader
    """

    tr_n, val_n = train_set_split(PATH, TRAIN, TEST, SEGMENTATION, exclude_list)
    seg_df = pd.read_csv(SEGMENTATION).set_index("ImageId")

    train_items = [os.path.join(TRAIN, f) for f in tr_n]
    val_items = [os.path.join(TRAIN, f) for f in val_n]
    all_items = train_items + val_items

    val_set = set(val_n)

    def is_valid(item_path):
        return os.path.basename(item_path) in val_set

    splits = FuncSplitter(is_valid)(all_items)

    def bbox_label_func(x):
        return label_func(os.path.basename(x), seg_df, height=sz, width=sz, num_queries=num_queries)

    dblock = DataBlock(
        blocks=(ImageBlock, (BBoxBlock, BBoxLblBlock)),
        get_items=noop,
        splitter=IndexSplitter(splits[1]),
        get_x=lambda x: x,
        get_y=bbox_label_func,
        item_tfms=Resize(sz),
        batch_tfms=aug_transforms(
            do_flip=False,
            flip_vert=False,
            max_rotate=0,
            max_lighting=0.05
        )
    )

    dls = dblock.dataloaders(all_items, bs=bs, num_workers=4)

    global test_names
    test_items = [os.path.join(TEST, f) for f in test_names]
    test_dl = dls.test_dl(test_items)

    return dls, test_dl
