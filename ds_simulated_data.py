import copy
import glob
import os
import random
import re
import warnings
from os.path import basename, join, splitext

import cv2
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import config
from DLBio.ds_pt_dataset import SegmentationDataset
from DLBio.helpers import find_image, is_match, search_in_all_subfolders
from helpers import load_image, load_label
# save some time and set to False, but if new images are added this should be
# set to true at least once
DO_RUN_DATA_CHECK = False
MAX_LOAD_SIZE = 100  # -> train data .9*200 = 180

# using imagenet normalization
NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


def get_dataloader(
    ds_type,
    batch_size, num_workers,
    ds_len=config.DATASET_LENGTH,
    use_only=None, use_rgb=None, use_pyr_down=None,
    num_classes=config.NUM_CLASSES,
    crop_size=512, aug_type='minimal'
):
    #assert use_rgb is not None

    data_aug = get_data_augmentation(crop_size, type_=aug_type)
    dataset = MinProjectionsBig(
        ds_type,
        data_aug=data_aug,
        ds_len=ds_len,
        num_classes=num_classes,
        use_only=use_only,
        use_rgb=use_rgb,
        use_pyr_down=use_pyr_down,
        normalize=NORMALIZE
    )
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_data_augmentation(crop_size, type_='minimal'):
    if type_ == 'minimal':
        data_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(crop_size),
        ])

    return data_aug


class MinProjectionsBig(SegmentationDataset):
    """Dataset that loads only half of the images into RAM. On each epoch,
    the data are randomly split into two sets. The first set is loaded. After,
    the first set is done, the second set is loaded.

    """

    def __init__(self,
                 ds_type,
                 data_aug=None, use_only=None, use_rgb=None, use_pyr_down=None, max_load_size=MAX_LOAD_SIZE,
                 **kwargs
                 ):
        super(MinProjectionsBig, self).__init__(**kwargs)

        assert use_rgb is not None
        self.use_rgb = use_rgb
        assert use_pyr_down is not None
        self.use_pyr_down = use_pyr_down

        # which images are used: all in the folder, or the ones specified
        paths_ = get_all_images(ds_type, None)
        assert paths_, f'no data found!'
        if use_only is None:
            use_only = paths_
        assert set(use_only).issubset(set(paths_))

        # create list of image and label paths
        self.x_paths = copy.deepcopy(use_only)
        self.y_paths = []

        # if there are more than N images they are loaded in batches
        num_divisions = len(self.x_paths) // max_load_size + 1
        print(
            f'num_divisions: {len(self.x_paths)}/{max_load_size}={num_divisions}'
        )
        self.nd = num_divisions

        # grab labels
        self.is_manual_label = False
        if ds_type == 'original':
            raise NotImplementedError()
        elif ds_type == 'simulation':
            label_folder = config.SIM_LABELS
            self.is_manual_label = True

        label_files = search_in_all_subfolders(r'.*.png', label_folder)
        assert label_files
        for x in use_only:
            y = find_image(x, label_files)
            if y is None:
                warnings.warn(f'No label for {x}')
                self.x_paths.remove(x)
                continue

            self.y_paths.append(y)

        self.x_paths = np.array(self.x_paths)

        assert self.y_paths, 'no fitting labels found in {label_folder}'

        self.y_paths = np.array(self.y_paths)

        # odd set number does not work that great with halving the whole set
        if self.y_paths.shape[0] % self.nd == 1:
            warnings.warn(f'Number of images not divisible by {self.nd}.')

        self.num_images = len(self.x_paths)

        assert data_aug is not None
        self.data_aug = data_aug

        if DO_RUN_DATA_CHECK:
            self._run_data_check()
        else:
            self._ran_data_check = True

        # set intials to these values to start a new splitting of the data
        # keep track of how often a batch was requested, if >= len/2
        # load the second set
        self.ctr = len(self) // self.nd
        # which one of the halves is currently loaded
        self.run_index = self.nd - 1

        self.current_split = None

    def _run_data_check(self):
        print('running data check...')
        # run data test
        self._set_split()
        for i in range(self.nd):
            print(f'...for division {i}')
            self.run_index = i
            self._load_data()
            super(MinProjectionsBig, self)._run_data_check()

    def __getitem__(self, index):
        # if this case is true, the current batch of data is processed and a
        # new one needs to be loaded
        if self.ctr >= len(self) // self.nd:
            # when the epoch is through, compute a new data-split for the next
            # epoch
            if self.run_index == self.nd - 1:
                self._set_split()
                self.run_index = 0

            # load a part of the set into RAM
            self._load_data()
            self.run_index += 1
            self.ctr = 0

        #
        self.ctr += 1

        # not all datasets need to be splitted and loaded again
        if self.nd == 1:
            self.ctr = 0
        return super(MinProjectionsBig, self).__getitem__(index)

    def _load_data(self):
        print('loading data...')
        self.images = list()
        self.labels = list()
        # load the images according to the current_hale
        # for i in tqdm(self.current_split[self.run_index]):
        for i in self.current_split[self.run_index]:
            self.images.append(load_image(
                self.x_paths[i], use_rgb=self.use_rgb, do_pyr_down=self.use_pyr_down))
            self.labels.append(load_label(
                self.y_paths[i], do_pyr_down=self.use_pyr_down,
                is_manual_label=self.is_manual_label
            ))

    def _set_split(self):
        # create random indeces to split the set in two halves
        n = self.num_images // self.nd
        rp = np.random.permutation(self.num_images)

        self.current_split = list()
        for i in range(self.nd):
            self.current_split.append(
                list(rp[i * n:(i + 1) * n])
            )


def get_all_images(ds_type, images_=None):
    if images_ is None:
        if ds_type == 'original':
            raise NotImplementedError()
        elif ds_type == 'simulation':
            images_ = search_in_all_subfolders(r'.*.png', config.SIM_IMAGES)

        else:
            raise ValueError('Unknown ds_type: {ds_type}')

        assert images_, 'no images found'
        images_ = sorted(images_)

    return images_


def get_image_paths(ds_type, split_type_, perc_split=.8, seed=0, images_=None):
    if images_ is None:
        assert split_type_ == 'split', 'Nothing is done here.'

    images_ = get_all_images(ds_type, images_)

    if split_type_ == 'all':
        return images_

    if split_type_ == 'split':
        print(f'split seed: {seed}')
        assert seed >= 0

        old_state = random.getstate()
        random.seed(seed)
        n_test = int(len(images_) * (1. - perc_split))
        test_data = []
        for _ in range(n_test):
            test_data.append(random.choice(images_))

        train = set(images_) - set(test_data)

        random.setstate(old_state)
        return sorted(list(train)), sorted(list(test_data))

# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------


def crop_dataset():
    j = 0
    ind = [15, 19, 35, 39, 59, 79, 99, 119, 139, 159, 179, 199]
    x_cor = [900, 2000, 750, 1200, 1100, 1900, 900, 1500, 900, 1100, 500, 1050]
    y_cor = [450, 650, 600, 650, 650, 300, 600, 500, 100, 200, 200, 650]
    h = 4700
    w = 4800
    paths_images = sorted(glob.glob(join(config.IMAGES_PRE, '*.bmp')))
    paths_labels = sorted(glob.glob(join(config.LABELS_PRE, '*.png')))
    for i, elem in enumerate(paths_labels):
        image = cv2.imread(paths_images[i])
        label = cv2.imread(paths_labels[i])
        if(i > ind[j]):
            j += 1
            print(i, j)
        image = image[y_cor[j]:y_cor[j] + h, x_cor[j]:x_cor[j] + w]
        label = label[y_cor[j]:y_cor[j] + h, x_cor[j]:x_cor[j] + w]
        cv2.imwrite(join(config.IMAGES, basename(paths_images[i])), image)
        cv2.imwrite(join(config.LABELS, basename(paths_labels[i])), label)


def _debug():
    from DLBio.pytorch_helpers import cuda_to_numpy
    from DLBio.helpers import to_uint8_image
    import matplotlib.pyplot as plt
    data_loader = get_dataloader(
        'simulation', 8, 0, use_rgb=True, use_pyr_down=False
    )
    for sample in data_loader:
        x = sample['x'].cpu().detach()
        y = sample['y'].cpu().detach()
        print(x.shape)
        batch_size = x.shape[0]
        for b in range(batch_size):
            tmp_x = to_uint8_image(cuda_to_numpy(x[b, ...]))
            tmp_y = np.array(y[b, ...])
            #_, ax = plt.subplots(1, 2)
            # ax[0].imshow(tmp_x)
            # ax[1].imshow(tmp_y)
            plt.figure(figsize=(15, 15))
            plt.imshow(tmp_x)
            plt.imshow(tmp_y, alpha=.7)

            plt.savefig('debug.png')
            plt.close()
            xxx = 0


if __name__ == "__main__":
    _debug()
