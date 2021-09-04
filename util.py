
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F
import torch.nn as nn

def get_auc_ap_score(labels, preds):

    auc_score = roc_auc_score(labels, preds)
    ap_score = average_precision_score(labels, preds)

    return auc_score, ap_score


def diceCoeff(pred, gt, smooth=1, activation='sigmoid'):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
 
    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = nn.Softmax2d()
    else:
        raise NotImplementedError("Activation implemented for sigmoid and softmax2d 激活函数的操作")
 
    pred = activation_fn(pred)
 
    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)
 
    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
 
    return loss.sum() / N