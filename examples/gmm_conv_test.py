# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 15:42:48 2020

@author: dm0051
"""

#from torch_geometric.nn import GMMConv
from gmm_conv2 import GMMConv as GMMConv2
import torch 

#net = GMMConv2(21, 10, 2, 24, separate_gaussians = False, flow = 'target_to_source')

net2 = GMMConv2(21, 10, 2, 24, separate_gaussians = False, flow = 'target_to_source')


from torch_scatter import scatter_mean
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

dataset = TUDataset(root='../tmp/ENZYMES', name='ENZYMES', use_node_attr=True)

input = dataset[0]

x = input.x
edge_index = input.edge_index
eij = torch.rand(168)
pseudo = torch.rand(168,2)

#gmm_out = net(x, edge_index, pseudo)
gmm_out2 = net2(x, edge_index, pseudo, eij)