import importlib

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv

class GcnNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(in_channels, 40)
        self.bn = torch.nn.BatchNorm1d(40)
        self.conv2 = GCNConv(40, 1)

        self.fc1 = torch.nn.Linear(num_nodes, 50)
        self.fc2 = torch.nn.Linear(50, out_channels)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr[:, 0]
        epsilon = 1e-10

        h = self.conv1(x, edge_index, edge_attr)
        h = F.leaky_relu(h)
        h = self.bn(h)
        h = self.conv2(h, edge_index, edge_attr)
        h = F.leaky_relu(h)
    
        out = h.view(data.num_graphs, -1)
        out = (out - torch.mean(out, dim = 1, keepdim = True)) / (torch.var(out, dim = 1, keepdim = True) + epsilon)

        out = self.fc1(out)
        out = self.relu(out)
        out = F.dropout(out, p = 0.5, training = self.training)
        out = self.fc2(out)

        return out