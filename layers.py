import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv, GCNConv

import networkx as nx
import numpy as np
import scipy.linalg


class SymmetricLinear(nn.Module):
    def __init__(self, size):
        super(SymmetricLinear, self).__init__()
        self.size = size
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.bias = nn.Parameter(torch.Tensor(size))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # Make the weight matrix symmetric
        w_symmetric = (self.weight + self.weight.t()) / 2
        return torch.nn.functional.linear(x, w_symmetric, self.bias)


class ManyBodyMPNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, max_order, edge_feature_dim, K=3):
        super(ManyBodyMPNNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_order = max_order
        self.K = K
        self.U, self.V = self.precompute_motif_eigenvectors(max_order + 1)

        self.W_cheb = nn.ParameterList([nn.Parameter(torch.randn(out_channels, in_channels, K)) for _ in range(max_order)])
        self.W_cheb_motif = nn.ParameterList([nn.Parameter(torch.randn(in_channels, k, K)) for k in range(2, max_order+1)])
        self.W_msg = nn.Linear(in_channels, out_channels)

    @staticmethod
    def precompute_motif_eigenvectors(n):
        U, V = {}, {}
        for i in range(2, n + 1):
            motif_graph = nx.star_graph(i - 1)
            laplacian_matrix = nx.laplacian_matrix(motif_graph).toarray()
            eigenvalues, eigenvectors = scipy.linalg.eigh(laplacian_matrix)
            U[i] = torch.linalg.inv(torch.tensor(eigenvectors, dtype=torch.float))
            V[i] = torch.tensor(eigenvalues, dtype=torch.float)
        return U, V

    def chebyshev_expansion(self, x, K):
        T_0, T_1 = torch.ones_like(x), x
        expansion = [T_0, T_1]
        for k in range(2, K + 1):
            T_k = 2 * x * T_1 - T_0
            expansion.append(T_k)
            T_0, T_1 = T_1, T_k
        return torch.stack(expansion, dim=-1)
    
    def apply_chebyshev(self, L, X, K):
        X_0 = X
        X_1 = torch.matmul(L, X)
        result = self.W_cheb[0][:, :, 0].unsqueeze(1) * X_0 + self.W_cheb[0][:, :, 1].unsqueeze(1) * X_1
        for k in range(2, K):
            X_k = 2 * torch.matmul(L, X_1) - X_0
            result += self.W_cheb[0][:, :, k].unsqueeze(1) * X_k
            X_0, X_1 = X_1, X_k
        return result

    def higher_order_interaction(self, H, edge_index):
        # Initialize the output feature matrix
        Y = torch.zeros_like(H)
        
        # Iterate through orders from 2 to max_order
        for order in range(2, self.max_order + 1):
            # Retrieve the precomputed eigenvectors U_k and eigenvalues for the current order
            U_k = self.U[order].to(H.device)
            lambda_k = self.V[order].to(H.device)

            # Scale the eigenvalues for Chebyshev expansion and perform the expansion
            scaled_eigenvalues = 2 * lambda_k / lambda_k.max() - 1
            T_k = self.chebyshev_expansion(scaled_eigenvalues, self.K)
            
            # Apply the filter defined by the Chebyshev expansion
            for i in range(self.K):
                # Chebyshev polynomial of degree i on the Laplacian, weighted by W_cheb
                filter_response = torch.matmul(U_k, torch.matmul(torch.diag(T_k[:, i]), U_k.t()))
                weights = self.W_cheb_motif[order-2][:, :, i]  # Assuming this needs to align with `result`
                result = torch.einsum("ni,ik->ni", H, torch.matmul(weights, filter_response.t()))
                Y += result

        return Y

    def forward(self, H, edge_index, edge_features=None):
        # Compute Laplacian
        edge_index, edge_weight = add_self_loops(edge_index, num_nodes=H.size(0))
        edge_weight = torch.ones(edge_index.size(1), device=H.device)
        L = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=torch.float, num_nodes=H.size(0))

        L_dense = to_dense_adj(L[0], edge_attr=L[1], max_num_nodes=H.size(0)).squeeze(0)
        L_scaled = 2 * L_dense - torch.eye(L_dense.size(0), device=H.device)

        # Direct interaction via a custom Chebyshev operation
        X_2 = self.apply_chebyshev(L_scaled, H, self.K)
        # Higher-order interactions
        m = self.higher_order_interaction(H, edge_index)
        # Combine direct and higher-order interactions
        H_next = X_2.transpose(0, 1).sum(-1) + self.W_msg(m)
        return H_next

    def __initialize_weights(self):
        for w in self.W_cheb:
            init.kaiming_uniform_(w, a=np.sqrt(5))
        for w in self.W_cheb_motif:
            init.kaiming_uniform_(w, a=np.sqrt(5))

