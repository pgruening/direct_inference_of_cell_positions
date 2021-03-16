import argparse
import glob
import os
import re
import shutil
import subprocess
import warnings
from os.path import basename, join, splitext, dirname
import re
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import config
import utils
from DLBio.helpers import MyDataFrame, check_mkdir, search_in_all_subfolders
from run_eval_time_lapse import (count_hits, dice_score, do_debug_plot,
                                 load_image, load_label, pre_process)
from eval_functions import count_hits, dice_score, do_debug_plot, load_prediction, pre_process
import warnings
# ----------------------------------------------------------------------------
# assumes folder structure to look like this:
# base -> {exp_folder0, ...} -> {type_folder0,...}
# e.g.
# base
#   - exp0
#       - holograms
#           - image0
#           - image1
#       - holographic_min_projections (exp0_ is removed in dst folder)
#           - image0
#           - image1
#       - cell_labels
#   - exp1
#       - holograms
#           - ...
# ----------------------------------------------------------------------------


# NOTE: set false if you use phase-min labels
# is the label a phase-min-image that needs thresholding or a binary image
USE_BINARY_LABEL = True

# dataset paths: parent folder is called BASE
# BASE = config.SIM_BASE
BASE = config.SIM_EVAL_BASE
# path to the training images
TRAIN_IMAGES = config.SIM_IMAGES

#BASELINE_THRES = config.THRESHOLD
BASELINE_THRES = 100

# where is the model saved that you want to use for inference
MODEL_BASE = config.EXP_FOLDER
MODEL_PATHS = {
    'resnet18': join(MODEL_BASE, 'smp_resnet18_sim15032021', 'model.pt'),
}

FILE_GLOB = '*.png'

# where to save any debug images
DO_SAVE_DEBUG_FIGS = True

# distinction whether baseline is used or one of the CNNs.
# in {'baseline', 'prediction'}
CURRENT_PREDICTION = None


def get_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segment_first', action='store_true')
    parser.add_argument('--base_path', type=str, default=BASE)
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--device', type=int, default=0)
    return parser.parse_args()


def segment_data(base, model, device):
    exp_folders_ = next(os.walk(base))[1]

    pred_folder = 'predictions_' + model

    model_path = MODEL_PATHS[model]

    for folder in exp_folders_:
        call_str = [
            'python', 'run_segment_folder.py',
            '--image_folder', join(base, folder, 'holograms'),
            '--out_folder', join(base, folder, pred_folder),
            '--model_path', model_path,
            '--file_glob', FILE_GLOB,
            '--device', str(device)
        ]

        subprocess.call(call_str)


def run(base_path, model):
    df = MyDataFrame()

    pred_paths = search_in_all_subfolders(
        _create_search_rgx(base_path, CURRENT_PREDICTION, model=model),
        base_path, match_on_full_path=True
    )

    label_paths = search_in_all_subfolders(
        _create_search_rgx(base_path, 'labels'),
        base_path, match_on_full_path=True
    )

    pred_paths = pred_paths[::1]

    for i, pred_path in tqdm(enumerate(pred_paths)):

        # load prediction
        if CURRENT_PREDICTION == 'baseline':
            # is phase min -> load like a label
            pred = load_label(pred_path, is_manual_label=False,
                              thres=BASELINE_THRES)
        else:
            # uint image [0, 255] -> [0, 1] cell prob
            pred = load_image(pred_path)

        pred = pre_process(pred)

        # load label
        gt_path = find_in_folder(pred_path, label_paths, base_path, model)
        if gt_path is None:
            warnings.warn(f'Could not find label for {pred_path}')
        try:
            gt = load_label(gt_path, is_manual_label=USE_BINARY_LABEL)
        except:
            warnings.warn(f'Could not load label {gt}')

        gt = pre_process(gt)
        if gt.sum() == 0:
            continue

        stats = {'name': get_id(pred_path, base_path, model)}
        do_continue = False
        for func in [count_hits, dice_score]:

            tmp = func(pred, gt)

            if tmp is None:
                do_continue = True
                warnings.warn(f'No cells found {pred_path}')
                break
            else:
                stats.update(tmp)

        if do_continue:
            continue

        df.update(stats)

        if DO_SAVE_DEBUG_FIGS:
            parent_folder = '/'.join(pred_path.split('/')[:-2])
            out_path = join(
                parent_folder, f'debug_dice_{model}', basename(pred_path)
            )

            utils.dice_plot(pred, gt, out_path, stats['dice'])

    df = df.get_df()
    out_file = join(base_path, 'results_' + model + '.xlsx')
    check_mkdir(out_file)

    df.to_excel(out_file)


def _create_search_rgx(base, type_, model=None):
    if type_ == 'predictions':
        assert model is not None

    # exp_folder + type folder
    folder = '(.*)' + {
        'predictions': f'predictions_{model}',
        'labels': 'cell_labels',
        'baseline': 'baseline_reconstructions'
    }[type_]

    return re.compile(join(base, folder, '(.*).(png|bmp)'))


def find_in_folder(x, paths_, base, model):
    # [base]/[exp]/predictions/[im_num].png
    # vs.
    # [base]/[exp]/labels/[im_num].png
    rgx_x = _create_search_rgx(base, CURRENT_PREDICTION, model)
    rgx_y = _create_search_rgx(base, 'labels')

    def _is_match(x, y):
        tmp = re.match(rgx_x, x)
        id_x = tmp.group(1) + tmp.group(2)
        tmp = re.match(rgx_y, y)
        id_y = tmp.group(1) + tmp.group(2)
        return id_x == id_y

    out = [z for z in paths_ if _is_match(x, z)]
    if not out:
        return None
    return out[0]


def get_id(x, base, model):
    rgx_x = _create_search_rgx(base, CURRENT_PREDICTION, model)
    tmp = re.match(rgx_x, x)
    return tmp.group(1) + tmp.group(2)


if __name__ == "__main__":
    global CURRENT_PREDICTION
    OPTIONS = get_options()
    if CURRENT_PREDICTION == 'baseline':
        OPTIONS.model = 'baseline'

    print(f'running for model: {OPTIONS.model}')

    if OPTIONS.segment_first:
        segment_data(OPTIONS.base_path, OPTIONS.model, OPTIONS.device)

    run(OPTIONS.base_path, OPTIONS.model)
