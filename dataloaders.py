import os
import torchvision.transforms as tfs
import torch.utils.data
import numpy as np
from PIL import Image
import random

def get_data_loaders(cfgs):
    batch_size = cfgs.get('batch_size', 64)
    image_size = cfgs.get('image_size', 64)
    crop = cfgs.get('crop', None)

    run_train = cfgs.get('run_train', False)
    train_val_data_dir = cfgs.get('train_val_data_dir', './data')
    run_test = cfgs.get('run_test', False)
    test_data_dir = cfgs.get('test_data_dir', './data/test')

    load_gt_depth = cfgs.get('load_gt_depth', False)
    AB_dnames = cfgs.get('paired_data_dir_names', ['A', 'B'])
    AB_fnames = cfgs.get('paired_data_filename_diff', None)

    train_loader = val_loader = test_loader = None
    if load_gt_depth:
        get_loader = lambda **kargs: get_paired_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    else:
        get_loader = lambda **kargs: get_image_loader(**kargs, batch_size=batch_size, image_size=image_size, crop=crop)

    if run_train:
        train_data_dir = os.path.join(train_val_data_dir, "train")
        val_data_dir = os.path.join(train_val_data_dir, "val")
        assert os.path.isdir(train_data_dir), "Training data directory does not exist: %s" %train_data_dir
        assert os.path.isdir(val_data_dir), "Validation data directory does not exist: %s" %val_data_dir
        print(f"Loading training data from {train_data_dir}")
        train_loader = get_loader(data_dir=train_data_dir, is_validation=False)
        print(f"Loading validation data from {val_data_dir}")
        val_loader = get_loader(data_dir=val_data_dir, is_validation=True)
    if run_test:
        assert os.path.isdir(test_data_dir), "Testing data directory does not exist: %s" %test_data_dir
        print(f"Loading testing data from {test_data_dir}")
        test_loader = get_loader(data_dir=test_data_dir, is_validation=True)

    return train_loader, val_loader, test_loader


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', 'webp')
def is_image_file(filename):
    return filename.lower().endswith(IMG_EXTENSIONS)


## simple image dataset ##
def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if 'val' in dir:
        rand_num = 10
    else:
        rand_num = 100

    images = []

    '''PIE'''
    session_list = os.listdir(dir)
    for session in session_list:
        object_list = os.listdir(os.path.join(dir, session))
        for object in object_list:
            expression_list = os.listdir(os.path.join(dir, session, object))
            for expression in expression_list:
                left_list = os.listdir(os.path.join(dir, session, object, expression, '13_0'))
                frontal_list = os.listdir(os.path.join(dir, session, object, expression, '05_1'))
                right_list = os.listdir(os.path.join(dir, session, object, expression, '04_1'))


                for i in range(rand_num):
                    left = random.choice(left_list)
                    frontal = random.choice(frontal_list)
                    right = random.choice(right_list)

                    left_path = os.path.join(dir, session, object, expression, '13_0', left)
                    frontal_path = os.path.join(dir, session, object, expression, '05_1', frontal)
                    right_path = os.path.join(dir, session, object, expression, '04_1', right)
                    images.append([frontal_path, right_path])
                    images.append([frontal_path, left_path])

    random.shuffle(images)
    return images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False):
        super(ImageDataset, self).__init__()
        if is_validation:
            self.root = os.path.join(data_dir, 'val')
        else:
            self.root = os.path.join(data_dir, 'train')
        self.paths = make_dataset(self.root)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, imgs):
        img1 = tfs.functional.resize(imgs[0], (self.image_size, self.image_size))
        img2 = tfs.functional.resize(imgs[1], (self.image_size, self.image_size))

        img1_tensor = tfs.functional.to_tensor(img1)
        img2_tensor = tfs.functional.to_tensor(img2)

        return img1_tensor, img2_tensor

    def __getitem__(self, index):
        fpath = self.paths[index % self.size]
        img1 = Image.open(fpath[0]).convert('RGB')
        img2 = Image.open(fpath[1]).convert('RGB')
        return self.transform([img1, img2])

    def __len__(self):
        return self.size

    def name(self):
        return 'ImageDataset'


def get_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None):

    dataset = ImageDataset(data_dir, image_size=image_size, crop=crop, is_validation=is_validation)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader


def make_paied_dataset(dir, AB_dnames=None, AB_fnames=None):
    A_dname, B_dname = AB_dnames or ('A', 'B')
    dir_A = os.path.join(dir, A_dname)
    dir_B = os.path.join(dir, B_dname)
    assert os.path.isdir(dir_A), '%s is not a valid directory' % dir_A
    assert os.path.isdir(dir_B), '%s is not a valid directory' % dir_B

    images = []
    for root_A, _, fnames_A in sorted(os.walk(dir_A)):
        for fname_A in sorted(fnames_A):
            if is_image_file(fname_A):
                path_A = os.path.join(root_A, fname_A)
                root_B = root_A.replace(dir_A, dir_B, 1)
                if AB_fnames is not None:
                    fname_B = fname_A.replace(*AB_fnames)
                else:
                    fname_B = fname_A
                path_B = os.path.join(root_B, fname_B)
                if os.path.isfile(path_B):
                    images.append((path_A, path_B))
    return images


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, image_size=256, crop=None, is_validation=False, AB_dnames=None, AB_fnames=None):
        super(PairedDataset, self).__init__()
        self.root = data_dir
        self.paths = make_paied_dataset(data_dir, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
        self.size = len(self.paths)
        self.image_size = image_size
        self.crop = crop
        self.is_validation = is_validation

    def transform(self, img, hflip=False):
        if self.crop is not None:
            if isinstance(self.crop, int):
                img = tfs.CenterCrop(self.crop)(img)
            else:
                assert len(self.crop) == 4, 'Crop size must be an integer for center crop, or a list of 4 integers (y0,x0,h,w)'
                img = tfs.functional.crop(img, *self.crop)
        img = tfs.functional.resize(img, (self.image_size, self.image_size))
        if hflip:
            img = tfs.functional.hflip(img)
        return tfs.functional.to_tensor(img)

    def __getitem__(self, index):
        path_A, path_B = self.paths[index % self.size]
        img_A = Image.open(path_A).convert('RGB')
        img_B = Image.open(path_B).convert('RGB')
        hflip = not self.is_validation and np.random.rand()>0.5
        tensor_A = self.transform(img_A, hflip=hflip)
        tensor_B = self.transform(img_B, hflip=hflip)
        return tensor_A, tensor_A.flip(-1), tensor_B

    def __len__(self):
        return self.size

    def name(self):
        return 'PairedDataset'


def get_paired_image_loader(data_dir, is_validation=False,
    batch_size=256, num_workers=4, image_size=256, crop=None, AB_dnames=None, AB_fnames=None):

    dataset = PairedDataset(data_dir, image_size=image_size, crop=crop, \
        is_validation=is_validation, AB_dnames=AB_dnames, AB_fnames=AB_fnames)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not is_validation,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader
