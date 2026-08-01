"""
Convert model outputs to binary masks and compute segmentation metrics.
Author: Sudipta Paul, email: pauls5@rpi.edu
"""


import torch

def logits_to_mask(logits):
    return torch.argmax(
        logits, dim=1, keepdim=True,
    ) == 1


def segmentation_metrics(prediction, target, epsilon=1e-5):
    prediction = prediction.bool()
    target = target.bool()
    dimensions = tuple(range(1, prediction.ndim))
    true_positive = (prediction & target).sum(dim=dimensions).float()
    false_positive = (prediction & ~target).sum(dim=dimensions).float()
    false_negative = (~prediction & target).sum(dim=dimensions).float()
    dice = (2.0 * true_positive + epsilon) / (2.0 * true_positive + false_positive + false_negative + epsilon)
    iou = (true_positive + epsilon) / (true_positive  + false_positive + false_negative + epsilon)
    precision = (true_positive + epsilon) / (true_positive  + false_positive + epsilon)
    recall = (true_positive + epsilon) / (true_positive + false_negative + epsilon)
    return dice, iou, precision, recall


def dice_iou(prediction, target, epsilon=1e-5):
    dice, iou, _, _ = segmentation_metrics(prediction,target,epsilon)
    return dice, iou
