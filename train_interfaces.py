import config
import torch
import numpy as np
import torch.nn as nn
from DLBio.pt_training import ITrainInterface

EVAL_THRES = 0.5


class BinarySegmentation(ITrainInterface):
    def __init__(self, model):
        self.model = model

        self.xent_loss = nn.CrossEntropyLoss()
        self.metrics = {
            'acc': accuracy,
            'dice': dice_score
        }

    def train_step(self, sample):
        images, targets = sample['x'].cuda(), sample['y'].cuda()
        pred = self.model(images)

        loss = self.xent_loss(pred, targets)
        return loss, {k: v(pred, targets) for k, v in self.metrics.items()}


def accuracy(y_pred, y_true):
    _, y_pred = y_pred.max(1)  # grab class predictions
    return (y_pred == y_true).float().mean().item()


def dice_score(y_pred, y_true):
    assert y_pred.shape[1] == 2
    assert y_true.max() <= 1.

    y_pred = torch.softmax(y_pred, 1)[:, 1, ...]

    y_true = y_true.float()

    y_pred = (y_pred > .5).float()

    true_pos = (y_true * y_pred).sum()

    false_pos = (y_pred - y_true).clamp(0, 1.).sum()
    false_neg = (y_true - y_pred).clamp(0, 1.).sum()

    dice = 2. * true_pos / (false_pos + false_neg + 2. * true_pos + 1e-9)

    return dice.item()


def apply_thres(pred):
    pred_thres = (pred > EVAL_THRES).astype('float32')
    return pred_thres
