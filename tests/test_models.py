import sys 
sys.path.append("./")

import functools
import torch 
import torch.nn as nn
import impl.models as models

from unittest import TestCase
from impl.models import MLP

torch.manual_seed(0)

hidden_dim = 64
conv_layer = 1
dropout = 0
jk_var = 1
z_ratio = 0.95
max_deg, output_channels = 4997, 3
pool="size"
aggr="sum"

class TestMLP(TestCase):
    def test_forward(self):
        testMLP = MLP(2, 2, 2, 1)
        sample = torch.tensor([1.0, 1.0], dtype=torch.float)
        out = testMLP.forward(sample).detach()
        req = torch.tensor([0.1017, -0.9128])
        assert torch.sum(out-req) < 1e-4

class TestGNN(TestCase):
    def load_model(self):
        conv = models.EmbZGConv(hidden_dim,
                        hidden_dim,
                        conv_layer,
                        max_deg=max_deg,
                        activation=nn.ELU(inplace=True),
                        jk=1,
                        dropout=dropout,
                        conv=functools.partial(models.GLASSConv,
                                                   aggr=aggr,
                                                   z_ratio=z_ratio,
                                                   dropout=dropout),
                        
                        gn=True)
        mlp = nn.Linear(hidden_dim * (conv_layer) if jk_var else hidden_dim,
                    output_channels)
        pool_fn_fn = {
                "mean": models.MeanPool,
                "max": models.MaxPool,
                "sum": models.AddPool,
                "size": models.SizePool
            }
        if pool in pool_fn_fn:
            pool_fn1 = pool_fn_fn[pool]()
        else:
            raise NotImplementedError

        gnn = models.GLASS(conv, torch.nn.ModuleList([mlp]),
                            torch.nn.ModuleList([pool_fn1]))
        gnn.load_state_dict(torch.load('density.pt'))
        assert gnn is not None 