import importlib

import torch
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.nn import GCNConv

class GcnNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, **kwargs):
        super(GcnNet, self).__init__()
        self.conv11 = GCNConv(in_channels, 64)
        self.conv12 = GCNConv(64, 1)
        #self.conv21 = GCNConv(in_channels, 64)
        #self.conv22 = GCNConv(64, 1)
        #self.conv3 = GCNConv(2, 1)
        self.fc1 = torch.nn.Linear(num_nodes, 128)
        self.fc2 = torch.nn.Linear(128, out_channels)
        self.relu = torch.nn.LeakyReLU()
    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        h1 = self.conv11(x, edge_index, edge_attr[:, 0])
        h1 = F.leaky_relu(h1)
        h1 = self.conv12(h1, edge_index, edge_attr[:, 0])
        h = F.leaky_relu(h1)

        """
        h2 = self.conv21(x, edge_index, edge_attr[:, 1])
        h2 = F.leaky_relu(h2)
        h2 = self.conv22(h2, edge_index, edge_attr[:, 1])
        h2 = F.leaky_relu(h2)

        h = torch.cat([h1, h2], dim=1)
        h = self.conv3(h, edge_index)
        """

        out = h.view(data.num_graphs, -1)

        out = self.fc1(out)
        out = self.relu(out)
        #out = F.dropout(out, p = 0.2, training = self.training)
        out = self.fc2(out)

        out = F.softmax(out, dim=1)

        return out