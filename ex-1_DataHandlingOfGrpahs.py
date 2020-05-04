#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 15:46:40 2019

@author: dipu
"""

import torch
from torch_geometric.data import Data

edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index)

print(data.keys)

for key, item in data:
    print('{} found in {}'.format(key,item))
    
#data.num_nodes
#data.num_edges
#data.num_node_features
#data.contains_isolated_nodes()    
#data.contains_self_loops()
#data.is_directed()

#%%    
edge_index = torch.tensor([[0, 1],
                       [1, 0],
                       [1, 2],
                       [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index.t().contiguous())

#%%
print(data.keys)
print(data.x)
print(data['x'])