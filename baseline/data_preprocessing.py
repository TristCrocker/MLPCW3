import os

import pandas as pd
from sklearn.model_selection import train_test_split

from fastai.vision.all import *


def train_set_split(PATH, TRAIN, TEST, SEGMENTATION, exclude_list):
    """
    Splits the train folder into train/val sets (5% val) and removes excluded files.
    """
    train_names = [f for f in os.listdir(TRAIN)]
    global test_names
    test_names = [f for f in os.listdir(TEST)]
    
    for el in exclude_list:
        if el in train_names:
            train_names.remove(el)
        if el in test_names:
            test_names.remove(el)

    # 5% of data in the validation set
    tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)
    
    return tr_n, val_n


class pdFilesDataset:
    """
    Legacy placeholder for old Fast.ai v0.7 'FilesDataset'.
    Not used in Fast.ai v2, but defined for compatibility.
    """
    def __init__(self, fnames, path, transform):
        self.fnames = fnames
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        return None  # No-op

def get_data(sz, bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list):
    """
    Builds a DataLoaders object using Fast.ai v2's DataBlock API.

    Arguments:
      sz (int) : image size to resize
      bs (int) : batch size
      PATH (str) : root path (not mandatory in v2, but kept for naming)
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

    def label_func(fname):
        """
        Returns 0 if no ship, 1 if ship is present (based on EncodedPixels).
        fname is the file's basename (e.g. 'xxx.jpg').
        """
        if fname not in seg_df.index:
            # Not found in CSV => no ship
            return 0
        masks = seg_df.loc[fname, 'EncodedPixels']
        # If it's float -> NaN or no EncodedPixels => 0
        if isinstance(masks, float):
            return 0
        return 1

    # 4) Build lists of train/val image paths
    train_items = [os.path.join(TRAIN, f) for f in tr_n]
    val_items = [os.path.join(TRAIN, f) for f in val_n]
    all_items = train_items + val_items

    val_set = set(val_n)  # basenames in val set
    def is_valid(item_path):
        # item_path is e.g. /path/to/train/filename.jpg
        fname = os.path.basename(item_path)
        return fname in val_set

    splits = FuncSplitter(is_valid)(all_items)

    item_tfms = Resize(sz)  # single transform for resizing
    batch_tfms = aug_transforms(
        do_flip=True,
        flip_vert=True,
        max_rotate=20,
        max_lighting=0.05
    )

    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock),
        get_items=noop,           # We'll pass our item list directly
        splitter=IndexSplitter(splits[1]),
        get_x=lambda x: x,        # x is already a full path
        get_y=lambda x: label_func(os.path.basename(x)),
        item_tfms=item_tfms,
        batch_tfms=batch_tfms
    )

    dls = dblock.dataloaders(all_items, bs=bs)

    global test_names
    test_items = [os.path.join(TEST, f) for f in test_names]
    test_dl = dls.test_dl(test_items)

    return dls, test_dl