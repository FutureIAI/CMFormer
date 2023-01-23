import numpy as np
import torch


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


# added by hxx for iou calculation
def intersectionAndUnion(imPred, imLab, numClass):

    imPred = imPred * (imLab > 0)

    intersection = imPred * (imPred == imLab)
    area_intersection = torch.histc(intersection, bins=numClass, min=1, max=numClass)

    # Compute area union:
    area_pred = torch.histc(imPred, bins=numClass, min=1, max=numClass)
    area_lab = torch.histc(imLab, bins=numClass, min=1, max=numClass)
    area_union = area_pred + area_lab - area_intersection

    return area_intersection, area_union


def accuracy(preds, label):
    valid = (label > 0)  # hxx
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum


def macc(preds, label, num_class):
    a = torch.zeros(num_class)
    b = torch.zeros(num_class)
    for i in range(num_class):
        mask = (label == i + 1)
        a_sum = (mask * preds == i + 1).sum()
        b_sum = mask.sum()
        a[i] = a_sum
        b[i] = b_sum
    return a, b


# ----------------------------------------------------

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


if __name__ == '__main__':
    pre = torch.rand((1, 40, 480, 640))
    label = np.load(r"E:\Dataset\nyudv2\labels\1.npy")
