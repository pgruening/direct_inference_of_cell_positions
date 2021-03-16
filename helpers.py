import warnings

import cv2

import config
import numpy as np
from DLBio.helpers import check_mkdir


def load_image(x, use_rgb=True, do_pyr_down=False):
    image = cv2.imread(x)
    if not use_rgb:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # NOTE: images are cut to avoid swapping
    if do_pyr_down:
        image = cv2.pyrDown(image)
    return image


def load_label(path, thres=config.LABEL_THRES, do_pyr_down=False, is_manual_label=False, type_='uint8'):
    assert isinstance(thres, int)
    assert thres > 0 and thres < 255
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if do_pyr_down:
        image = cv2.pyrDown(image)

    # the manually generated labels are already thresholded
    if is_manual_label:
        label = (image > 0).astype(type_)
    else:
        label = (image < thres).astype(type_)

    if not label.max() == 1. and label.min() == 0.:
        print(image.min(), image.mean(), image.max())
        warnings.warn(f'problems with label: {path}')

    #assert label.max() == 1, f'no positive class in image {x}'
    return label


def dice_plot(pred, ground_truth, out_path, dice, phase_min=None, alpha=.1):
    if phase_min is not None:
        if phase_min.ndim == 2:
            phase_min = cv2.cvtColor(phase_min, cv2.COLOR_GRAY2RGB)

    out_image = np.zeros(list(pred.shape) + [3]).astype('uint8')

    assert pred.min() == 0. and pred.max() == 1.
    assert ground_truth.min() == 0. and ground_truth.max() == 1.

    tp_image = pred * ground_truth > 0
    fp_image = pred - ground_truth > 0
    fn_image = ground_truth - pred > 0

    pxl_color = {
        'tp': (255, 255, 255),  # white
        'fn': (255, 0, 0),  # red
        'fp': (0, 0, 255)  # blue
    }

    for i in range(3):
        out_image[fp_image, i] = pxl_color['fp'][i]
        out_image[fn_image, i] = pxl_color['fn'][i]
        out_image[tp_image, i] = pxl_color['tp'][i]

    if phase_min is not None:
        alpha_chan = 1. - (out_image.sum(-1) == 0).astype('float32')
        a = np.ones(alpha_chan.shape)
        a[alpha_chan > 0] = alpha
        a = a[..., np.newaxis]
        b = ((1. - alpha) * alpha_chan)[..., np.newaxis]
        out_image = a * phase_min + b * out_image
        out_image = out_image.astype('uint8')

    cv2.putText(
        out_image,  # numpy array on which text is written
        f'{dice:.3f}',  # text
        (100, 100),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        4,  # font size
        (255, 0, 0),  # font color
        3
    )

    check_mkdir(out_path)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_image)
