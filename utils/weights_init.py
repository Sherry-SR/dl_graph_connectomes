import importlib

import torch
import torch_geometric.nn as gnn
import torch.nn.init as init

def weights_init(m):
    if isinstance(m, gnn.GCNConv):
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0)
    if isinstance(m, torch.nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)