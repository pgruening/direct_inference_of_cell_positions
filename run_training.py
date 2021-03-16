import glob
import json
import os
from os.path import join

import config
import DLBio.pt_training as pt_training
import numpy as np
import torch
from DLBio.helpers import check_mkdir, find_image, save_options
from DLBio.pt_train_printer import Printer

from model_getter import get_model
from train_interfaces import BinarySegmentation

# increases the number of iterations for the validation set
VAL_DATASET_MULT = 10
# After this number of batches, the training status is printed to the terminal
PRINT_FREQ = 20


def get_options():
    parser = pt_training.get_train_arg_parser(config)
    parser.add_argument('--seg_model', type=str, default=None)
    parser.add_argument('--dataset', type=str, default='no_artifact')
    parser.add_argument('--freeze_enc', action='store_true')

    parser.add_argument('--perc_split', type=float, default=config.PERC_SPLIT)
    parser.add_argument('--split_seed', type=int, default=config.SPLIT_SEED)
    parser.add_argument('--aug_type', type=str, default=config.AUG_TYPE)

    # rgb flag added - default false
    parser.add_argument('--use_rgb', action='store_true')
    parser.add_argument('--use_pyr_down', action='store_true')

    return parser.parse_args()


def run(options):
    if options.device is not None:
        pt_training.set_device(options.device)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    pt_training.set_random_seed(options.seed)

    folder = join(config.EXP_FOLDER, options.folder)
    check_mkdir(folder, is_dir=True)

    save_options(join(folder, 'seg_opt.json'), options)

    _train_model(options, folder, device)


def _train_model(options, folder, device):

    model_name = os.path.splitext(options.model_name)[0]

    model_out = join(folder, model_name + '.pt')
    log_file = join(folder, f'seg_log_{model_name}.json')

    if options.seg_model is not None:
        load_model_path = join(folder, options.seg_model)
        print(f'loading model: {load_model_path}')
        model_sd = torch.load(load_model_path).state_dict()
        model = get_model(options.model_type, options.in_dim,
                          options.num_classes, device=device)
        model.load_state_dict(model_sd, strict=False)
    else:
        model = get_model(options.model_type, options.in_dim,
                          options.num_classes, device=device)

    if options.freeze_enc:
        model.freeze_encoder()

    optimizer = pt_training.get_optimizer(
        options.opt, model.parameters(), options.lr, momentum=options.mom)

    scheduler = None
    if options.lr_steps > 0:
        scheduler = pt_training.get_scheduler(
            options.lr_steps, options.epochs, optimizer)

    dl_train, dl_test, early_stopping = setup_dataset(options)

    train_interface = get_train_interface(options, model)

    #pt_training.loss_verification(train_interface, dl_train, Printer(100))

    training = pt_training.Training(
        optimizer, dl_train, train_interface,
        scheduler=scheduler, printer=Printer(PRINT_FREQ, log_file),
        save_path=model_out, save_steps=options.sv_int,
        val_data_loader=dl_test, early_stopping=early_stopping
    )

    training(options.epochs)


def to_uint8_image(image):
    """Rescale and cast image to uint8 format
    so it can be written to a png or jpg file
    Parameters
    ----------
    image : numpy array
      Image to transform
    Returns
    -------
    numpy array of type uint8
      Image is rescaled to [0,255] and casted.
    """
    if image.dtype == 'uint8':
        return image

    image -= np.min(image)
    image /= np.max(image)
    return (255 * image).astype('uint8')


def get_train_interface(options, model):
    train_interface = BinarySegmentation(model)

    return train_interface


def setup_dataset(options):
    import ds_simulated_data

    # split into two image paths, save those data to a json file
    im_train, im_test = ds_simulated_data.get_image_paths(
        options.dataset,
        'split',
        perc_split=options.perc_split,
        seed=options.split_seed
    )

    out_splits = join(config.EXP_FOLDER, options.folder, 'split.json')
    check_mkdir(out_splits)
    with open(out_splits, 'w') as file:
        json.dump({'train': im_train, 'test': im_test}, file)

    early_stopping = None

    # if early stopping is used, further split the training set to get a
    # validation set
    if options.early_stopping:
        # test on subset of training set
        im_train, im_test = ds_simulated_data.get_image_paths(
            options.dataset,
            'split', perc_split=options.perc_split, images_=im_train
        )

        val_splits = join(config.EXP_FOLDER, options.folder, 'val_split.json')
        check_mkdir(val_splits)
        with open(val_splits, 'w') as file:
            json.dump({'train': im_train, 'val': im_test}, file)

        assert options.sv_int == -1
        early_stopping = pt_training.EarlyStopping(
            options.es_metric, get_max=True, epoch_thres=options.epochs // 5
        )

    # note that dl_train and dl_test are the training and validation dataloaders
    # dl_test is used for early stopping
    # If you want to further test your model, grab the test image names in the
    # json file.
    dl_train = ds_simulated_data.get_dataloader(
        options.dataset,
        options.bs,
        options.nw,
        crop_size=options.crop_size,
        ds_len=options.ds_len,
        aug_type=options.aug_type,
        num_classes=options.num_classes,
        use_only=im_train,
        use_rgb=options.use_rgb,
        use_pyr_down=options.use_pyr_down
    )

    dl_test = ds_simulated_data.get_dataloader(
        options.dataset,
        options.bs,
        options.nw,
        crop_size=options.crop_size,
        ds_len=len(im_test) * VAL_DATASET_MULT,
        aug_type=options.aug_type,
        num_classes=options.num_classes,
        use_only=im_test,
        use_rgb=options.use_rgb,
        use_pyr_down=options.use_pyr_down
    )

    return dl_train, dl_test, early_stopping


if __name__ == "__main__":
    OPTIONS = get_options()
    run(OPTIONS)
