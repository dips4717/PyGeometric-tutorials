#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 09:15:06 2019

@author: dipu
"""

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
dataset.num_classes
dataset.num_edge_attributes
dataset.num_edge_features



data = dataset[0]
data.num_edges
#data.num_attr
data.num_nodes


for i, batch in enumerate(loader):
    print('\n',i)
    print(batch)
    print(batch.num_graphs)
    

'''batch is a column vector which maps each node to its respective graph in the batch:

batch=[0⋯01⋯n−2n−1⋯n−1]⊤
You can use it to, e.g., average node features in the node dimension for each graph individually:
'''
from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for i, data in enumerate(loader):
    print(data)
    print(data.num_graphs)
    x = scatter_mean(data.x, data.batch, dim=0)
    print(x.size())
    if i ==0:
        break