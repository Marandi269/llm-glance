from unittest import TestCase

import torch
from torch import Tensor
import torch.nn.functional as F

from attention import SelfAttention
from transformer import get_y


class AttentionTest(TestCase):
    def test_attention(self):
        attention = SelfAttention(16, 4)
        print(attention)

    def test_contiguous(self):
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])