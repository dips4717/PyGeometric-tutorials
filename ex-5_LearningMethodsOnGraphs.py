#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 10:28:03 2019

@author: dipu
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

dataset = Planetoid(root='tmp/Cora', name='Cora')


#class GCNConv(MessagePassing):
#    def __init__(self, in_channels, out_channels):
#        super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation.
#        self.lin = torch.nn.Linear(in_channels, out_channels)
#
#    def forward(self, x, edge_index):
#        # x has shape [N, in_channels]
#        # edge_index has shape [2, E]
#
#        # Step 1: Add self-loops to the adjacency matrix.
#        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
#
#        # Step 2: Linearly transform node feature matrix.
#        x = self.lin(x)
#
#        # Step 3-5: Start propagating messages.
#        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
#
#    def message(self, x_j, edge_index, size):
#        # x_j has shape [E, out_channels]
#
#        # Step 3: Normalize node features.
#        row, col = edge_index
#        deg = degree(row, size[0], dtype=x_j.dtype)
#        deg_inv_sqrt = deg.pow(-0.5)
#        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
#
#        return norm.view(-1, 1) * x_j
#
#    def update(self, aggr_out):
#        # aggr_out has shape [N, out_channels]
#
#        # Step 5: Return new node embeddings.
#        return aggr_out

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    
    
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()    
    
    
model.eval()
_, pred = model(data).max(dim=1)
correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / data.test_mask.sum().item()
print('Accuracy: {:.4f}'.format(acc))
