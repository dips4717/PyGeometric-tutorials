# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:40:13 2020
GMMConv Layer from PyGeometric modified to take eij parameter as implemented in SGRN paper.


@author: dm0051
"""

import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
import math 

#from ..inits import zeros, glorot

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)

EPS = 1e-15


class GMMConv(MessagePassing):
    r"""The gaussian mixture model convolutional operator from the `"Geometric
    Deep Learning on Graphs and Manifolds using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \frac{1}{|\mathcal{N}(i)|}
        \sum_{j \in \mathcal{N}(i)} \frac{1}{K} \sum_{k=1}^K
        \mathbf{w}_k(\mathbf{e}_{i,j}) \odot \mathbf{\Theta}_k \mathbf{x}_j,

    where

    .. math::
        \mathbf{w}_k(\mathbf{e}) = \exp \left( -\frac{1}{2} {\left(
        \mathbf{e} - \mathbf{\mu}_k \right)}^{\top} \Sigma_k^{-1}
        \left( \mathbf{e} - \mathbf{\mu}_k \right) \right)

    denotes a weighting function based on trainable mean vector
    :math:`\mathbf{\mu}_k` and diagonal covariance matrix
    :math:`\mathbf{\Sigma}_k`.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        dim (int): Pseudo-coordinate dimensionality.
        kernel_size (int): Number of kernels :math:`K`.
        separate_gaussians (bool, optional): If set to :obj:`True`, will
            learn separate GMMs for every pair of input and output channel,
            inspired by traditional CNNs. (default: :obj:`False`)
        aggr (string, optional): The aggregation operator to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"mean"`)
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, dim, kernel_size,
                 separate_gaussians=False, aggr='mean', root_weight=True,
                 bias=True, **kwargs):
        super(GMMConv, self).__init__(aggr=aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dim = dim
        self.kernel_size = kernel_size
        self.separate_gaussians = separate_gaussians

        self.g = Parameter(
            torch.Tensor(in_channels, out_channels * kernel_size))

        if not self.separate_gaussians:
            self.mu = Parameter(torch.Tensor(kernel_size, dim))
            self.sigma = Parameter(torch.Tensor(kernel_size, dim))
        else:
            self.mu = Parameter(
                torch.Tensor(in_channels, out_channels, kernel_size, dim))
            self.sigma = Parameter(
                torch.Tensor(in_channels, out_channels, kernel_size, dim))

        if root_weight:
            self.root = Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.g)
        glorot(self.mu)
        glorot(self.sigma)
        glorot(self.root)
        zeros(self.bias)

    def forward(self, x, edge_index, pseudo, eij):
        """
        inject e_ij here, and pass them into message as well.
        """
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = pseudo.unsqueeze(-1) if pseudo.dim() == 1 else pseudo

        N, K, M = x.size(0), self.kernel_size, self.out_channels

        if not self.separate_gaussians:
            out = torch.matmul(x, self.g).view(N, K, M)
            out = self.propagate(edge_index, x=out, pseudo=pseudo, eij=eij)
        else:
            out = self.propagate(edge_index, x=x, pseudo=pseudo, eij=eij)

        if self.root is not None:
            out = out + torch.matmul(x, self.root)

        if self.bias is not None:
            out = out + self.bias

        return out

    def message(self, x_j, pseudo, eij):
        """
        x_j (): index_select of x; indices used edge_index[0]. shape (num_edges, K, num_out_channel)  
        pseudo (tensor): Pseudo-coordinates of shape (Num_edges, dim)
        """
        F, M = self.in_channels, self.out_channels
        (E, D), K = pseudo.size(), self.kernel_size

        if not self.separate_gaussians:
            gaussian = -0.5 * (pseudo.view(E, 1, D) -
                               self.mu.view(1, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, K]

            #return (x_j.view(E, K, M) * gaussian.view(E, K, 1)).sum(dim=-2)  # [E,M] 
            return (x_j.view(E, K, M) * gaussian.view(E, K, 1) * eij.view(E,1,1)).sum(dim=-2)  # [E,M]
            

        else:
            gaussian = -0.5 * (pseudo.view(E, 1, 1, 1, D) -
                               self.mu.view(1, F, M, K, D)).pow(2)
            gaussian = gaussian / (EPS + self.sigma.view(1, F, M, K, D).pow(2))
            gaussian = torch.exp(gaussian.sum(dim=-1))  # [E, F, M, K]

            gaussian = gaussian * self.g.view(1, F, M, K)
            gaussian = gaussian.sum(dim=-1)  # [E, F, M]

            return (x_j.view(E, F, 1) * gaussian).sum(dim=-2)  # [E, M]

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
