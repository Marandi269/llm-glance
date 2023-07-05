import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    def __init__(self, k, heads=4, mask=False):
        super().__init__()

        assert k % heads == 0
        self.k, self.heads, self.mask = k, heads, mask
        self.to_keys = nn.Linear(k, k, bias=False)
        self.to_queries = nn.Linear(k, k, bias=False)
        self.to_values = nn.Linear(k, k, bias=False)

        self.unify_heads = nn.Linear(k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.heads

        queries = self.to_queries(x)
        keys = self.to_keys(x)
        values = self.to_values(x)

        s = k // h
        keys = keys.view(b, t, h, s)
        queries = queries.view(b, t, h, s)
        values = values.view(b, t, h, s)

        # - fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, s)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, s)
        values = values.transpose(1, 2).contiguous().view(b * h, t, s)

        # Get dot product of queries and keys, and scale
        dot = torch.bmm(queries, keys.transpose(1, 2))  # -- dot has size (b*h, t, t) containing raw weights
        dot = dot / (k ** (1 / 2))  # scale the dot product
        dot = F.softmax(dot, dim=2)  # normalize, dot now contains row-wise normalized weights

        out = torch.bmm(dot, values).view(b, h, t, s)  # apply the self attention to the values

        # swap h, t back, unify heads
        out = out.transpose(1, 2).contiguous().view(b, t, s * h)

        return self.unifyheads(out)
