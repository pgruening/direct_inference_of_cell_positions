import warnings

import cv2
import numpy as np
from DLBio.rectangles import TopLeftRectangle

import config

DO_DEBUG_RECTANGLES = False


def dice_score(pred, ground_truth):
    assert pred.min() >= 0. and pred.max() <= 1.
    assert ground_truth.min() >= 0. and ground_truth.max() <= 1.
    intersection = (pred * ground_truth).sum()
    union = (pred + ground_truth).clip(max=1.).sum()
    union = max(1., union)

    return {'dice': intersection / union}


def phase_min_pixel_values(pred, ground_truth, phase_min):
    out = {}
    pred_vals = phase_min[pred > 0].flatten()
    gt_vals = phase_min[ground_truth > 0].flatten()

    for perc in [50, 75, 95]:
        out[f'pred_pxl_{perc}'] = np.percentile(pred_vals, perc)
        out[f'gt_pxl_{perc}'] = np.percentile(gt_vals, perc)

    return out


def count_hits(pred, ground_truth):
    assert pred.min() >= 0. and pred.max() <= 1.
    assert ground_truth.min() >= 0. and ground_truth.max() <= 1.

    # get rectangles around connected components
    rect_p = get_rectangle_array(pred)
    #ground_truth = get_rectangle_array(ground_truth)
    rect_gt = get_rectangle_array(ground_truth)

    if rect_gt is None:
        warnings.warn('No cells found for Ground truth')
        return None

    if rect_p is None:
        warnings.warn('No cells found for Prediction')
        return None

    # returns Matrix of shape num_pred x num_gt
    rect_ious = estimate_rect_iou(rect_p, rect_gt)
    out = greedy_match(rect_ious, rect_p, rect_gt)
    return out


def greedy_match(rect_ious, pred, gt, match_thres=config.MATCH_THRES):
    num_predictions = rect_ious.shape[0]
    num_ground_truths = rect_ious.shape[1]

    unmatched_pred = list(range(num_predictions))
    unnmatched_gt = list(range(num_ground_truths))

    # try to find a match for each ground truth cell
    for i in range(num_ground_truths):
        if not unnmatched_gt:
            continue
        tmp = np.argmax(rect_ious[unmatched_pred, i])
        index = unmatched_pred[tmp]
        if rect_ious[index, i] >= match_thres:
            unmatched_pred.remove(index)
            unnmatched_gt.remove(i)

    # predictions = true_positives + false_positives
    false_positives = len(unmatched_pred)
    true_positives = num_predictions - false_positives

    # ground_truth = true_positives + false_negatives
    false_negatives = num_ground_truths - true_positives

    # look which kind of cells are not detected (area-wise...)
    out = {
        'tps': true_positives,
        'fps': false_positives,
        'fns': false_negatives,
        'num_pred_cells': true_positives + false_positives,
        'num_gt_cells': true_positives + false_negatives
    }

    out.update({
        'precision': true_positives / (true_positives + false_positives),
        'recall': true_positives / (true_positives + false_negatives),
    })

    out['precision'] = max(out['precision'], 1e-9)
    out['recall'] = max(out['recall'], 1e-9)

    f1_score = 2. * out['precision'] * out['recall']
    if f1_score < 1e-9:
        f1_score = 0.

    f1_score = f1_score / (out['precision'] + out['recall'])
    out.update({
        'f1_score': f1_score
    })

    # check areas for different types of detections
    w_pred = pred[:, cv2.CC_STAT_WIDTH]
    h_pred = pred[:, cv2.CC_STAT_HEIGHT]

    w_gt = gt[:, cv2.CC_STAT_WIDTH]
    h_gt = gt[:, cv2.CC_STAT_HEIGHT]

    area_all = np.concatenate([w_pred * h_pred, w_gt * h_gt], 0).mean()
    if len(unmatched_pred) > 0:
        area_fps = (w_pred[unmatched_pred] * h_pred[unmatched_pred]).mean()
    else:
        area_fps = -1.
    if len(unnmatched_gt) > 0:
        area_fns = (w_gt[unnmatched_gt] * h_gt[unnmatched_gt]).mean()
    else:
        area_fns = -1.

    out.update(
        {
            'area_all': area_all,
            'area_fps': area_fps,
            'area_fns': area_fns
        }
    )

    return out


def estimate_rect_iou(pred, ground_truth):
    X0 = pred[:, cv2.CC_STAT_LEFT]
    X1 = ground_truth[:, cv2.CC_STAT_LEFT]

    # left = max(x0, x1)
    left = _compute_for_all_pairs(X0, X1, lambda x: np.max(x, -1))

    Y0 = pred[:, cv2.CC_STAT_TOP]
    Y1 = ground_truth[:, cv2.CC_STAT_TOP]
    # top = max(y0, y1)
    top = _compute_for_all_pairs(Y0, Y1, lambda x: np.max(x, -1))

    # right = min(x0 + w0, x1 + w1)
    W0 = pred[:, cv2.CC_STAT_WIDTH]
    W1 = ground_truth[:, cv2.CC_STAT_WIDTH]
    right = _compute_for_all_pairs(X0 + W0, X1 + W1, lambda x: np.min(x, -1))

    # bottom = min(y0 + h0, y1 + h1)
    H0 = pred[:, cv2.CC_STAT_HEIGHT]
    H1 = ground_truth[:, cv2.CC_STAT_HEIGHT]
    bottom = _compute_for_all_pairs(Y0 + H0, Y1 + H1, lambda x: np.min(x, -1))

    # a = max(right - left, 0)
    # b = max(bottom - top, 0)
    A = (right - left).clip(min=0)
    B = (bottom - top).clip(min=0)

    # area_intersection = a * b
    intersection = A * B

    # union = W0 * H0 + W1 * H1 - intersection
    union = _compute_for_all_pairs(
        W0 * H0, W1 * H1, lambda x: x[..., 0] + x[..., 1])
    union = union - intersection

    # make sure to not divide by zero
    union[union == 0] = 1.

    rectangle_iou = intersection / union

    return rectangle_iou


def _compute_for_all_pairs(P, Q, func):
    NP = P.shape[0]
    NQ = Q.shape[0]

    P = P.reshape(-1, 1)
    Q = Q.reshape(1, -1)

    P = np.repeat(P, NQ, 1)
    Q = np.repeat(Q, NP, 0)

    tmp = np.stack([P, Q], -1)

    return func(tmp)


def get_rectangle_array(bin_image):
    rectangles = cv2.connectedComponentsWithStats(
        bin_image.astype('uint8'), 4, cv2.CV_32S)[2]

    # first rect is background
    if rectangles.shape[0] == 1:
        return None
    return rectangles[1:, ...]


def _to_rectangles(cc_stats):
    # cc_stats = cc_stats[2]

    out = []
    # assumes background component has been removed
    for i in range(0, cc_stats.shape[0]):
        tmp = cc_stats[i]
        x = tmp[cv2.CC_STAT_LEFT]
        y = tmp[cv2.CC_STAT_TOP]
        w = tmp[cv2.CC_STAT_WIDTH]
        h = tmp[cv2.CC_STAT_HEIGHT]

        out.append(TopLeftRectangle(x=x, y=y, w=w, h=h))

    return out


def load_prediction(path, prob_thres=.5):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    image = (image > prob_thres * 255).astype('float32')
    return image


def pre_process(image):
    K = config.OPENING_KERNEL
    kernel = np.ones((K, K))
    image = cv2.morphologyEx(
        image.astype('uint8'), cv2.MORPH_OPEN, kernel).astype('float32')
    return image
# ----------------------------------------------------------------------------
# -----------------------DEBUG & TESTS----------------------------------------
# ----------------------------------------------------------------------------


def do_debug_rectangles(pred, ground_truth, time_index, out_folder):
    if not DO_DEBUG_RECTANGLES:
        return

    #
    """
    kernel = np.ones((3, 3))
    pred = cv2.morphologyEx(
        pred.astype('uint8'), cv2.MORPH_OPEN, kernel).astype('float32')
    ground_truth = cv2.morphologyEx(
        ground_truth.astype('uint8'), cv2.MORPH_OPEN, kernel).astype('float32')
    """

    out_image = np.zeros(list(pred.shape) + [3]).astype('uint8')

    assert pred.min() >= 0. and pred.max() <= 1.
    assert ground_truth.min() >= 0. and ground_truth.max() <= 1.
    # draw pixels
    tp_image = pred * ground_truth > 0
    fp_image = pred - ground_truth > 0
    fn_image = ground_truth - pred > 0

    pxl_color = {
        'tp': (255, 255, 255),  # white
        'fn': (255, 0, 0),  # red
        'fp': (127, 127, 127)  # gray
    }

    for i in range(3):
        out_image[fp_image, i] = pxl_color['fp'][i]
        out_image[fn_image, i] = pxl_color['fn'][i]
        out_image[tp_image, i] = pxl_color['tp'][i]

    # get rectangles around connected components
    pred_rect_arr = get_rectangle_array(pred)
    gt_rect_arr = get_rectangle_array(ground_truth)

    rect_ious = estimate_rect_iou(gt_rect_arr, pred_rect_arr)

    # NOTE: copy from greedy match, keep up to date with original function
    not_matched = list(range(rect_ious.shape[1]))

    # keep for visualization
    matched = []
    unnmatched_gt = list(range(rect_ious.shape[0]))

    for i in range(rect_ious.shape[0]):
        tmp = np.argmax(rect_ious[i, not_matched])
        index = not_matched[tmp]
        iou = rect_ious[i, index]
        if iou >= config.MATCH_THRES:
            not_matched.remove(index)

            unnmatched_gt.remove(i)
            matched.append(index)
        else:
            # case: almost a match
            if iou > 0:
                a = pred_rect_arr[index, :4]
                b = gt_rect_arr[i, :4]

                a = TopLeftRectangle(
                    x=a[0], y=a[1],
                    w=a[cv2.CC_STAT_WIDTH],
                    h=a[cv2.CC_STAT_HEIGHT],
                )
                a = a.get_viewable(color=(0, 255, 0))
                a.add_cv_rectangle(out_image)

                b = TopLeftRectangle(
                    x=b[0], y=b[1],
                    w=b[cv2.CC_STAT_WIDTH],
                    h=b[cv2.CC_STAT_HEIGHT],
                )
                b = b.get_viewable(color=(0, 0, 255))
                #b.confidence = iou
                b.add_cv_rectangle(out_image)

            # no match found for this cell
            else:
                b = gt_rect_arr[i, :4]
                b = TopLeftRectangle(
                    x=b[0], y=b[1],
                    w=b[cv2.CC_STAT_WIDTH],
                    h=b[cv2.CC_STAT_HEIGHT],
                )
                b = b.get_viewable(color=(0, 0, 255))
                b.confidence = iou
                b.add_cv_rectangle(out_image)
                b.add_cv_rectangle(out_image)

    rect_color = {
        'tp': (0, 255, 0),  # green
        # 'fp': (255, 0, 255),  # purple
        'fp': (0, 255, 0),  # green
        # 'fn': (255, 0, 0),  # red
        'fn': (0, 0, 255),  # blue
        'gt': (0, 0, 255)  # blue
    }

    # draw rectangles
    gt_rect = _to_rectangles(gt_rect_arr)
    pred_rect = _to_rectangles(pred_rect_arr)

    # draw ground_truth
    for i, r in enumerate(gt_rect):
        if i in unnmatched_gt:
            r = r.get_viewable(color=rect_color['fn'])
            # r.add_cv_rectangle(out_image)

        else:
            r = r.get_viewable(color=rect_color['gt'])
            # max_iou = rect_ious[i, :].max()
            # r.confidence = max_iou
            # r.add_cv_rectangle(out_image)

    # draw matches and misses
    for i, r in enumerate(pred_rect):
        if i in matched:
            r = r.get_viewable(color=rect_color['tp'])
            # r.add_cv_rectangle(out_image)

        else:
            r = r.get_viewable(color=rect_color['fp'])
            max_iou = rect_ious[:, i].max()
            # if max_iou > 0:
            r.confidence = max_iou

            # seems to be working now
            #neighbors = _get_neighbors(r, gt_rect)
            # second_run = np.array([r.estimate_jaccard_index(x)
            #                       for x in gt_rect])
            #max_iou_2 = second_run.max()
            #index = np.argmax(second_run)
            # if max_iou_2 != max_iou:
            #    xxx = 0

            r.add_cv_rectangle(out_image, font_color=(255, 0, 0))

    out_path = join(out_folder, 'rectangles',
                    str(time_index).zfill(5) + '.png')
    check_mkdir(out_path)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_image)


def _get_neighbors(rect, other_rectangles, thres=10):
    out = []
    for r in other_rectangles:
        val = np.sqrt((r.x - rect.x)**2. + (r.y - rect.y)**2.)
        if val <= thres:
            out.append(r)
    return out


def do_debug_plot(pred, ground_truth, index, dice, out_folder):
    if not DO_DEBUG_PLOT:
        return

    out_image = np.zeros(list(pred.shape) + [3]).astype('uint8')

    assert pred.min() >= 0. and pred.max() <= 1.
    assert ground_truth.min() >= 0. and ground_truth.max() <= 1.

    tp_image = pred * ground_truth > 0
    fp_image = pred - ground_truth > 0
    fn_image = ground_truth - pred > 0

    pxl_color = {
        'tp': (255, 255, 255),  # white
        'fn': (255, 0, 0),  # red
        'fp': (127, 127, 127)  # gray
    }

    for i in range(3):
        out_image[fp_image, i] = pxl_color['fp'][i]
        out_image[fn_image, i] = pxl_color['fn'][i]
        out_image[tp_image, i] = pxl_color['tp'][i]

    plt.title(f'{dice:.3f}')

    cv2.putText(
        out_image,  # numpy array on which text is written
        f'{dice:.3f}',  # text
        (100, 100),  # position at which writing has to start
        cv2.FONT_HERSHEY_SIMPLEX,  # font family
        4,  # font size
        (255, 0, 0),  # font color
        3
    )

    out_path = join(out_folder, 'dice', str(index).zfill(5) + '.png')
    check_mkdir(out_path)
    out_image = cv2.cvtColor(out_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, out_image)


def _test_rectangle_method():
    R1 = [
        TopLeftRectangle(x=0, y=1, w=1, h=1),
        TopLeftRectangle(x=2, y=4, w=5, h=3),
        TopLeftRectangle(x=6, y=0, w=3, h=7)
    ]

    R2 = [
        TopLeftRectangle(x=2, y=4, w=5, h=3),
        TopLeftRectangle(x=6, y=0, w=3, h=7),
        TopLeftRectangle(x=0, y=1, w=1, h=1),
        TopLeftRectangle(x=10, y=10, w=1, h=1),
        TopLeftRectangle(x=2, y=4, w=2, h=2)
    ]

    out_gt = np.zeros((len(R1), len(R2)))

    for i, r1 in enumerate(R1):
        for j, r2 in enumerate(R2):
            out_gt[i, j] = r1.estimate_jaccard_index(r2)

    pred = _to_stat_output(R1)
    gt = _to_stat_output(R2)

    to_test = estimate_rect_iou(pred, gt)

    assert np.abs(to_test - out_gt).sum() < 1e-9
    print('Test ran successful')


def _test_f1_score():
    pred = [
        TopLeftRectangle(x=0, y=1, w=1, h=1),  # TP
        TopLeftRectangle(x=2, y=4, w=5, h=3),  # TP
        TopLeftRectangle(x=6, y=0, w=3, h=7),  # TP
        TopLeftRectangle(x=20, y=20, w=4, h=4),  # FP
        TopLeftRectangle(x=30, y=30, w=3, h=3)  # FP
    ]

    gt = [
        TopLeftRectangle(x=0, y=1, w=1, h=1),
        TopLeftRectangle(x=2, y=4, w=5, h=3),
        TopLeftRectangle(x=6, y=0, w=3, h=7),
        TopLeftRectangle(x=8, y=10, w=3, h=3),  # FN
        TopLeftRectangle(x=8, y=15, w=3, h=3)  # FN
    ]

    pred = _to_stat_output(pred)
    gt = _to_stat_output(gt)

    rect_ious = estimate_rect_iou(pred, gt)

    out = greedy_match(rect_ious, pred, gt)

    assert out['fps'] == 2
    assert out['fns'] == 2
    assert out['tps'] == 3

    print('test ran successful')


def _to_stat_output(R):
    X = [r.x for r in R]
    Y = [r.y for r in R]
    W = [r.w for r in R]
    H = [r.h for r in R]

    out = np.zeros((len(R), 4))
    out[:, cv2.CC_STAT_LEFT] = np.array(X)
    out[:, cv2.CC_STAT_TOP] = np.array(Y)
    out[:, cv2.CC_STAT_WIDTH] = np.array(W)
    out[:, cv2.CC_STAT_HEIGHT] = np.array(H)

    return out
