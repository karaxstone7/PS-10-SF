import numpy as np
import torch


###################       metrics      ###################
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict

    def clear(self):
        self.initialized = False


###################      cm metrics      ###################
class ConfuseMatrixMeter(AverageMeter):
    """Computes and stores the confusion matrix"""
    def __init__(self, n_class):
        super(ConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def harmonic_mean(xs):
    return len(xs) / sum((x+1e-6)**-1 for x in xs)


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)

    mean_F1 = np.nanmean(F1)
    return mean_F1


def cm2score(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    acc = tp.sum() / (hist.sum() + np.finfo(np.float32).eps)

    recall = tp / (sum_a1 + np.finfo(np.float32).eps)
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    mean_F1 = np.nanmean(F1)

    iu = tp / (sum_a1 + hist.sum(axis=0) - tp + np.finfo(np.float32).eps)
    mean_iu = np.nanmean(iu)

    freq = sum_a1 / (hist.sum() + np.finfo(np.float32).eps)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

    cls_iou = dict(zip(['iou_'+str(i) for i in range(hist.shape[0])], iu))
    cls_precision = dict(zip(['precision_'+str(i) for i in range(hist.shape[0])], precision))
    cls_recall = dict(zip(['recall_'+str(i) for i in range(hist.shape[0])], recall))
    cls_F1 = dict(zip(['F1_'+str(i) for i in range(hist.shape[0])], F1))

    score_dict = {'acc': acc, 'miou': mean_iu, 'mf1': mean_F1}
    score_dict.update(cls_iou)
    score_dict.update(cls_F1)
    score_dict.update(cls_precision)
    score_dict.update(cls_recall)
    return score_dict


###################      FIXED CONFUSION MATRIX      ###################
def get_confuse_matrix(num_classes, label_gts, label_preds):
    """Compute confusion matrix for batch"""

    def __fast_hist(label_gt, label_pred):
        # Convert to numpy if tensors
        if isinstance(label_gt, torch.Tensor):
            label_gt = label_gt.detach().cpu().numpy()
        if isinstance(label_pred, torch.Tensor):
            label_pred = label_pred.detach().cpu().numpy()

        label_gt = label_gt.flatten()
        label_pred = label_pred.flatten()

        mask = (label_gt >= 0) & (label_gt < num_classes)

        hist = np.bincount(
            num_classes * label_gt[mask].astype(int) + label_pred[mask],
            minlength=num_classes ** 2
        ).reshape(num_classes, num_classes)

        return hist

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for lt, lp in zip(label_gts, label_preds):
        confusion_matrix += __fast_hist(lt, lp)
    return confusion_matrix


def get_mIoU(num_classes, label_gts, label_preds):
    confusion_matrix = get_confuse_matrix(num_classes, label_gts, label_preds)
    score_dict = cm2score(confusion_matrix)
    return score_dict['miou']
