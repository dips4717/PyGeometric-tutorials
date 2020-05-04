#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:49:49 2019

@author: dipu
"""
import torch 

from torch_geometric.datasets import TUDataset

dataset = TUDataset(root='tmp/ENZYMES', name='ENZYMES')
dataset.num_classes
dataset.num_node_features
print('Length of the dataset {} is  {}'.format(dataset.name, len(dataset)))

#%%Acessing the each graph in the dataset
data = dataset[0]

#%% Suffling the dataset
dataset = dataset.shuffle()
#Or,
perm = torch.randperm(len(dataset))
dataset = dataset[perm]

#%% Cora dataset Example
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='tmp/Cora', name = 'Cora')
print('Length of the dataset {} is {}'.format(dataset.name, len(dataset)))
print('The number of classes in the dataset {} is {}'.format(dataset.name, dataset.num_classes))
print('The number of the node features is {}'.format(dataset.num_node_features))

## this data has a single and undirected citation graph
data = dataset[0]
print('datais \n',data)

data.train_mask.sum().item()
data.val_mask.sum().item()
data.test_mask.sum().item()