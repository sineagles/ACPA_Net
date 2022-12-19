import numpy as np
import torch
def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    pred = pred.view(-1).data.cpu().numpy()
    gt = gt.view(-1).data.cpu().numpy()

    tp = np.sum((pred==1)&(gt==1))
    fp = np.sum((pred==1)&(gt==0))
    fn = np.sum((pred==0)&(gt==1))
    tn = np.sum((pred==0)&(gt==0))

    return tp, fp, fn, tn


def dice(pred, gt, **kwargs):
    """2TP / (2TP + FP + FN)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float(2. * tp / (2 * tp + fp + fn+epsilon))


def iou(pred, gt, **kwargs):
    """TP / (TP + FP + FN)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float(tp / (tp + fp + fn +epsilon))


def precision(pred, gt, **kwargs):
    """TP / (TP + FP)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float(tp / (tp + fp  + epsilon))


def accuracy(pred, gt, **kwargs):    #ACC/AC
    """(TP + TN) / (TP + FP + FN + TN)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float((tp+tn) / (tp + fp +tn +fn+ epsilon))


def fscore(pred, gt, beta=1., **kwargs):
    """(1 + b^2) * TP / ((1 + b^2) * TP + b^2 * FN + FP)"""
    epsilon = 1.0e-6
    precision_ = precision(pred, gt)
    recall_ = recall(pred, gt)

    return (1 + beta*beta) * precision_ * recall_ /\
        ((beta*beta * precision_) + recall_+epsilon)


def sensitivity(pred, gt, **kwargs):  #SE TPR：true positive rate
    """TP / (TP + FN)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn= get_statistics(pred, gt)
    return float(tp / (tp + fn + epsilon))


def specificity(pred, gt, **kwargs):   #SP  TNR：true negative rate
    """TN / (TN + FP)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float(tn / (tp + fn + epsilon))


def recall(pred, gt, **kwargs):
    """TP / (TP + FN)"""
    return sensitivity(pred, gt)


def false_positive_rate(pred, gt, **kwargs):
    """FP / (FP + TN)"""
    return 1 - specificity(pred, gt)


def false_omission_rate(pred, gt, **kwargs):
    """FN / (TN + FN)"""
    epsilon = 1.0e-6
    tp, fp, fn, tn = get_statistics(pred, gt)
    return float(fn / (fn + tn + epsilon))


def false_negative_rate(pred, gt, **kwargs):
    """FN / (TP + FN)"""
    return 1 - sensitivity(pred, gt)


def true_negative_rate(pred, gt, **kwargs):
    """TN / (TN + FP)"""
    return specificity(pred, gt)


def false_discovery_rate(pred, gt, **kwargs):
    """FP / (TP + FP)"""
    return 1 - precision(pred, gt)


def negative_predictive_value(pred, gt, **kwargs):
    """TN / (TN + FN)"""
    return 1 - false_omission_rate(pred, gt)


def total_positives_test(pred, gt, **kwargs):
    """TP + FP"""
    tp, fp, fn, tn = get_statistics(pred, gt)
    return tp + fp


def total_negatives_test(pred, gt, **kwargs):
    """TN + FN"""
    tp, fp, fn, tn = get_statistics(pred, gt)
    return tn + fn


def total_positives_reference(pred, gt, **kwargs):
    """TP + FN"""
    tp, fp, fn, tn = get_statistics(pred, gt)
    return tp + fn


def total_negatives_reference(pred, gt, **kwargs):
    """TN + FP"""
    tp, fp, fn, tn = get_statistics(pred, gt)
    return tn + fp


ALL_METRICS = {
    "False Positive Rate": false_positive_rate,
    "Dice": dice,
    "Jaccard": iou,
    "Precision": precision,
    "Recall": recall,
    "Accuracy": accuracy,
    "specificity":  specificity,
    "sensitivity #SE": sensitivity,
    "False Omission Rate": false_omission_rate,
    "Negative Predictive Value": negative_predictive_value,
    "False Negative Rate": false_negative_rate,
    "True Negative Rate": true_negative_rate,
    "False Discovery Rate": false_discovery_rate,
    "Total Positives Test": total_positives_test,
    "Total Negatives Test": total_negatives_test,
    "Total Positives Reference": total_positives_reference,
    "total Negatives Reference": total_negatives_reference
}