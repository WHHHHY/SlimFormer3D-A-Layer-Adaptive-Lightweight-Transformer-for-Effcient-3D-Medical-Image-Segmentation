import math
import torch
import torch.nn as nn
import numpy as np

import sys
from lib.utils import *



class IoU(object):
    def __init__(self, num_classes=33, sigmoid_normalization=False):

        super(IoU, self).__init__()
        self.num_classes = num_classes
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):

        input = self.normalization(input)

        seg = torch.argmax(input, dim=1)
        assert seg.shape == target.shape, "dimensions of seg and target do not match"
        seg = seg.type(torch.uint8)
        target = target.type(torch.uint8)

        return intersect_and_union(seg, target, self.num_classes, reduce_zero_label=False)




if __name__ == '__main__':
    pred = torch.randn((4, 33, 32, 32, 16))
    gt = torch.randint(33, (4, 32, 32, 16))

    SO_metric = SurfaceOverlappingValues(num_classes=33, c=6, theta=1.0)

    SO_per_channel = SO_metric(pred, gt)

    print(SO_per_channel)




















