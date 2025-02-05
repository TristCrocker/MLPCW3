


def train_set_split(PATH, TRAIN, TEST, SEGMENTATION,
                    exclude_list):
    train_names = [f for f in os.listdir(TRAIN)]
    test_names = [f for f in os.listdir(TEST)]
    for el in exclude_list:
        if(el in train_names): train_names.remove(el)
        if(el in test_names): test_names.remove(el)
    #5% of data in the validation set is sufficient for model evaluation
    tr_n, val_n = train_test_split(train_names, test_size=0.05, random_state=42)

    return tr_n, val_n

class pdFilesDataset(FilesDataset):
    def __init__(self, fnames, path, transform):
        self.segmentation_df = pd.read_csv(SEGMENTATION).set_index('ImageId')
        super().__init__(fnames, transform, path)
    
    def get_x(self, i):
        img = open_image(os.path.join(self.path, self.fnames[i]))
        if self.sz == 768: return img 
        else: return cv2.resize(img, (self.sz, self.sz))
    
    def get_y(self, i):
        if(self.path == TEST): return 0
        masks = self.segmentation_df.loc[self.fnames[i]]['EncodedPixels']
        if(type(masks) == float): return 0 #NAN - no ship 
        else: return 1
    
    def get_c(self): return 2 #number of classes


def get_data(sz,bs, PATH, TRAIN, TEST, SEGMENTATION, exclude_list):
    tr_n, val_n = train_set_split(PATH, TRAIN, TEST, SEGMENTATION, exclude_list)
    #data augmentation
    aug_tfms = [RandomRotate(20, tfm_y=TfmType.NO),
                RandomDihedral(tfm_y=TfmType.NO),
                RandomLighting(0.05, 0.05, tfm_y=TfmType.NO)]
    tfms = tfms_from_model(arch, sz, crop_type=CropType.NO, tfm_y=TfmType.NO, 
                aug_tfms=aug_tfms)
    ds = ImageData.get_ds(pdFilesDataset, (tr_n[:-(len(tr_n)%bs)],TRAIN), 
                (val_n,TRAIN), tfms, test=(test_names,TEST))
    md = ImageData(PATH, ds, bs, num_workers=nw, classes=None)
    md.is_multi = False
    return md