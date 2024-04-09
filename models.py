import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ModuleList

from torch_geometric.nn import (
    ChebConv,
    GCNConv,
    SAGEConv,
    global_mean_pool,
)

from layers import ManyBodyMPNNConv


class BaseGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, conv_layer_type, dropout=0.5, *args, **kwargs):
        super(BaseGNN, self).__init__()
        self.convs = ModuleList()
        self.bns = ModuleList()
        self.dropout = dropout

        self.convs.append(conv_layer_type(in_channels, hidden_channels, *args, **kwargs))
        self.bns.append(BatchNorm1d(hidden_channels))
        for _ in range(1, num_layers - 1):
            self.convs.append(conv_layer_type(hidden_channels, hidden_channels, *args, **kwargs))
            self.bns.append(BatchNorm1d(hidden_channels))
        self.convs.append(conv_layer_type(hidden_channels, out_channels, *args, **kwargs))
        self.bns.append(BatchNorm1d(out_channels))

        self.lin1 = Linear(out_channels, 1)

        self.__initialize_weights()

    def forward(self, data):
        x, edge_index, batch = data.x.to(torch.float32), data.edge_index, data.batch
        edge_features = data.edge_attr.unsqueeze(1).to(torch.float32) if hasattr(data, 'edge_attr') else None

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_features) if edge_features is not None else conv(x, edge_index)
            x = self.bns[i](x)
            x = F.leaky_relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = global_mean_pool(x, batch)
        x = self.lin1(x)
        return x

    def __initialize_weights(self):
        for conv in self.convs:
            if hasattr(conv, 'weight'):
                nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu')
            if hasattr(conv, 'bias') and conv.bias is not None:
                nn.init.zeros_(conv.bias)
        nn.init.kaiming_uniform_(self.lin1.weight, nonlinearity='relu')
        nn.init.zeros_(self.lin1.bias)


class ManyBodyMPNN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, max_order, edge_feature_dim, K=3, dropout=0.5):
        super(ManyBodyMPNN, self).__init__(in_channels, hidden_channels, out_channels,
                                  num_layers, ManyBodyMPNNConv, dropout, max_order, edge_feature_dim, K)

class GCN(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GCN, self).__init__(in_channels, hidden_channels, out_channels, num_layers, GCNConv, dropout)

class GraphSage(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout=0.5):
        super(GraphSage, self).__init__(in_channels, hidden_channels, out_channels, num_layers, SAGEConv, dropout)

class ChebNet(BaseGNN):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, K, dropout=0.5):
        super(ChebNet, self).__init__(in_channels, hidden_channels, out_channels, num_layers, ChebConv, dropout, K)
