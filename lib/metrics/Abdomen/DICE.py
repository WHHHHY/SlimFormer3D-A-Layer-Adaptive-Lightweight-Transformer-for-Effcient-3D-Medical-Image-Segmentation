import torch
import torch.nn as nn

from lib.utils import *


class DICE(object):
    def __init__(self, num_classes=2, sigmoid_normalization=False, mode="extension"):
        super(DICE, self).__init__()
        self.num_classes = num_classes
        self.mode = mode
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)


    def __call__(self, input, target):

        target = expand_as_one_hot(target.long(), self.num_classes)

        assert input.dim() == target.dim() == 5, "After one-hot encoding, both prediction and label tensors are expected to be 5-dimensional."

        assert input.size() == target.size(), "After one-hot encoding, the prediction and label tensors must match in shape."

        input = self.normalization(input)

        return compute_per_channel_dice(input, target, epsilon=1e-6, mode=self.mode)





