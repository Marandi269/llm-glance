import torch
import torch.nn.functional as F
from torch import Tensor


def get_y(xs):
    raw_weights: Tensor = torch.bmm(xs, xs.transpose(1, 2))
    weights = F.softmax(raw_weights, dim=2)

    y = torch.bmm(weights, xs)
    return y
