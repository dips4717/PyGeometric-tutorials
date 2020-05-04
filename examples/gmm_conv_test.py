# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:42:48 2020

@author: dm0051
"""

#from torch_geometric.nn import GMMConv
from gmm_conv2 import GMMConv
import torch 


net = GMMConv(21, 10, 2, 24, separate_gaussians = False)


from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='../tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

input = dataset[0]

x = input.x
edge_index = input.edge_index
pseudo = torch.rand(168,2)

gmm_out = net(x, edge_index, pseudo)