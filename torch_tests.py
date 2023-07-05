from unittest import TestCase

import torch
from torch import Tensor
import torch.nn.functional as F

from transformer import get_y


class TorchTest(TestCase):
    def test_get_device_info(self):
        device = torch.device("mps")
        print(device)
        torch.set_default_device('mps')

    def test_vector_multiplication(self):
        x: Tensor = torch.ones([1, 2])
        print(x.transpose(0, 1).shape)
        print(x * x.transpose(0, 1))
        print(torch.mm(x, x.transpose(0, 1)))
        print('x:', x)
        print('x.transpose(0, 1):', x.transpose(0, 1))

    def test_bmm(self):
        xs: Tensor = torch.ones([3, 4, 5])

        raw_weights: Tensor = torch.bmm(xs, xs.transpose(1, 2))
        print(raw_weights)

        weights = F.softmax(raw_weights, dim=2)
        print('-' * 32)
        print('weights:', weights.shape)
        print(weights)

        y = torch.bmm(weights, xs)

        print('-' * 32)
        print(f'y: {y.shape}\n', y)
        print('-' * 32)
        print(f'weights[0]:{weights[0].shape}\n', weights[0])
        print('-' * 32)

        print(f'xs[0]:{xs[0].shape}\n', xs[0])
        print('-' * 32)

        print('weights[0]@xs[0]:\n', weights[0] @ xs[0])
        print('mm(weights[0],xs[0]):\n', torch.mm(weights[0], xs[0]))

    def test_get_y(self):
        xs: Tensor = torch.ones([3, 4, 5])
        print(get_y(xs))
