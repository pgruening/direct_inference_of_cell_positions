import argparse
import glob
from os.path import basename, join, splitext

import cv2
import numpy as np
import torch
from torchvision import transforms

import config
from DLBio.helpers import check_mkdir
#from DLBio.pt_model_class import CellSegmentationModel
from patchwise_inference import CellSegmentationObject
from DLBio.pt_training import set_device
from DLBio.pytorch_helpers import get_device
from ds_simulated_data import load_image, NORMALIZE


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--image_folder', type=str)
    parser.add_argument('--out_folder', type=str)
    parser.add_argument('--file_glob', type=str)
    parser.add_argument('--crop_size', type=int, default=config.CROP_SIZE)
    parser.add_argument('--device', type=int, default=None)
    return parser.parse_args()


def run(options):
    set_device(options.device)
    device = get_device()
    model = torch.load(options.model_path)
    model = model.eval().to(device)

    model_obj = CellSegmentationObject(
        device,
        model,
        [options.crop_size, options.crop_size],
        2,
        normalization=NORMALIZE
    )

    check_mkdir(options.out_folder)

    image_paths = glob.glob(join(options.image_folder, options.file_glob))
    assert image_paths, f'no images found at: {join(options.image_folder, options.file_glob)}'

    for im_path in image_paths:
        print(f'processing: {im_path}')
        id_ = splitext(basename(im_path))[0]

        # NOTE: possible error source, the default kwargs are used here
        image = load_image(im_path)

        pred = model_obj.do_task(image)[..., -1]

        assert pred.min() >= 0. and pred.max() <= 1., 'no softmax output'
        pred_image = (255. * pred).astype('uint8')

        cv2.imwrite(join(options.out_folder, id_ + '.png'), pred_image)


if __name__ == "__main__":
    OPTIONS = get_options()
    run(OPTIONS)
