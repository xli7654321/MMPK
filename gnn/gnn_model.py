import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool, global_mean_pool, global_max_pool, GlobalAttention
from torch_geometric.utils import add_self_loops, scatter, softmax
from torch_geometric.nn.inits import glorot, zeros
from .gnn_feature import get_atom_feat_dim, get_bond_feat_dim

atom_feat_dim = get_atom_feat_dim()
bond_feat_dim = get_bond_feat_dim()

class AtomEncoder(nn.Module):
    """Adapted from `ogb.graphproppred.mol_encoder`"""
    def __init__(self, d_emb):
        super().__init__()
        # list of Embedding layers for each atom feature
        self.atom_embedding_list = nn.ModuleList()

        for _, dim in enumerate(atom_feat_dim):
            embedding = nn.Embedding(dim, d_emb)
            nn.init.xavier_uniform_(embedding.weight.data)
            self.atom_embedding_list.append(embedding)

    def forward(self, x):
        x_embeddings = 0
        for i in range(x.shape[1]):
            # embedding feature i of all atoms
            x_embeddings += self.atom_embedding_list[i](x[:, i])

        return x_embeddings

class BondEncoder(nn.Module):
    """Adapted from `ogb.graphproppred.mol_encoder`"""
    def __init__(self, d_emb):
        super().__init__()
        # list of Embedding layers for each bond feature
        self.bond_embedding_list = nn.ModuleList()

        for _, dim in enumerate(bond_feat_dim):
            embedding = nn.Embedding(dim, d_emb)
            nn.init.xavier_uniform_(embedding.weight.data)
            self.bond_embedding_list.append(embedding)

    def forward(self, edge_attr):
        bond_embeddings = 0
        for i in range(edge_attr.shape[1]):
            # embedding feature i of all bonds
            bond_embeddings += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embeddings

class GINConv(MessagePassing):
    """Adapted from `pretrain-gnns.chem.model`"""
    def __init__(self, d_emb):
        super().__init__(aggr='add')
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, d_emb * 2),
            nn.ReLU(),
            nn.Linear(d_emb * 2, d_emb)
        )
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(d_emb)

    def forward(self, x, edge_index, edge_attr):
        # add self-loop
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # (2, num_edges + num_nodes)
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)),
                                     dtype=edge_attr.dtype, device=edge_attr.device)  # (num_nodes, num_edge_feats)
        self_loop_attr[:, 0] = 4  # index encoding of self-loop bond type
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.bond_encoder(edge_attr)  # (num_edges, d_emb)

        # propagate <- message + aggregate + update
        return self.mlp((1 + self.eps) * x + self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings))
    
    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out

class GCNConv(MessagePassing):
    def __init__(self, d_emb):
        super().__init__(aggr='add')
        self.linear = nn.Linear(d_emb, d_emb)
        self.bond_encoder = BondEncoder(d_emb)
    
    def forward(self, x, edge_index, edge_attr):
        # add self-loop
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # (2, num_edges + num_nodes)
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)),
                                     dtype=edge_attr.dtype, device=edge_attr.device)  # (num_nodes, num_edge_feats)
        self_loop_attr[:, 0] = 4  # index encoding of self-loop bond type
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        # edge_weight
        edge_weight = torch.ones((edge_index.size(1),), dtype=x.dtype, device=edge_index.device)
        
        # degree
        row, col = edge_index[0], edge_index[1]  # (2, num_edges) -> row: source nodes; col: target nodes
        deg = scatter(edge_weight, row, dim=0, dim_size=x.size(0), reduce='sum')  # weighted degree
        deg_inv_sqrt = deg.pow(-0.5)  # degree inverse square root
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        
        # normalization
        norm = edge_weight * deg_inv_sqrt[row] * deg_inv_sqrt[col]

        x = self.linear(x)  # (num_nodes, d_emb)
        edge_embeddings = self.bond_encoder(edge_attr)  # (num_edges, d_emb)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)
    
    def update(self, aggr_out):
        return aggr_out

class GATConv(MessagePassing):
    """Adapted from `pretrain-gnns.chem.model`
    Edge Attention, Self-loop as Node Attention"""
    def __init__(self, d_emb, num_heads=1, negative_slope=0.2):
        super().__init__(aggr='add')
        self.d_emb = d_emb
        self.num_heads = num_heads
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(d_emb, num_heads * d_emb)
        self.att = nn.Parameter(torch.Tensor(1, num_heads, d_emb * 2))
        self.bias = nn.Parameter(torch.Tensor(d_emb))

        self.bond_encoder = BondEncoder(num_heads * d_emb)

        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self-loop
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))  # (2, num_edges + num_nodes)
        self_loop_attr = torch.zeros((x.size(0), edge_attr.size(1)),
                                     dtype=edge_attr.dtype, device=edge_attr.device)  # (num_nodes, num_edge_feats)
        self_loop_attr[:, 0] = 4  # index encoding of self-loop bond type
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        
        x = self.weight_linear(x)  # (num_nodes, num_heads * d_emb)
        edge_embeddings = self.bond_encoder(edge_attr)  # (num_edges, d_emb)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_embeddings)
    
    def message(self, x_i, x_j, edge_index, edge_attr):
        x_i = x_i.view(-1, self.num_heads, self.d_emb)
        x_j = x_j.view(-1, self.num_heads, self.d_emb)
        edge_attr = edge_attr.view(-1, self.num_heads, self.d_emb)  # (num_edges, num_heads, d_emb)
        x_j += edge_attr

        # attention score
        alpha = (torch.cat((x_i, x_j), dim=-1) * self.att).sum(dim=-1)  # (num_edges, num_heads)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])  # edge_index[0]: source nodes
        alpha = alpha.view(-1, self.num_heads, 1)

        return (x_j * alpha).view(-1, self.num_heads * self.d_emb)
    
    def update(self, aggr_out):
        aggr_out = aggr_out.view(-1, self.num_heads, self.d_emb)
        aggr_out = aggr_out.mean(dim=1)
        aggr_out += self.bias

        return aggr_out

class NodeEncoder(nn.Module):
    """Adapted from `pretrain-gnns.chem.model`"""
    def __init__(self, num_layers, d_emb, JK='last', dropout=0., gnn_type='gin', num_heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.JK = JK
        self.dropout = dropout

        self.atom_encoder = AtomEncoder(d_emb)
        
        self.gnns = nn.ModuleList()
        for _ in range(num_layers):
            if gnn_type == 'gin':
                self.gnns.append(GINConv(d_emb))
            elif gnn_type == 'gcn':
                self.gnns.append(GCNConv(d_emb))
            elif gnn_type == 'gat':
                self.gnns.append(GATConv(d_emb, num_heads))
        
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.batch_norms.append(nn.BatchNorm1d(d_emb))

    def forward(self, *args):
        if len(args) == 3:
            x, edge_index, edge_attr = args[0], args[1], args[2]
        elif len(args) == 1:
            gdata = args[0]
            x, edge_index, edge_attr = gdata.x, gdata.edge_index, gdata.edge_attr
        else:
            raise ValueError('Invalid input arguments.')
        
        x_embeddings = self.atom_encoder(x)  # (num_nodes, d_emb)
        h_list = [x_embeddings]  # Initial node embeddings
        for layer in range(self.num_layers):
            # update node embeddings, keep edge embeddings unchanged
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layers - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)
            h_list.append(h)
        
        if self.JK == 'concat':
            node_embedding = torch.cat(h_list, dim=1)  # (num_nodes, (num_layers + 1) * d_emb)
        elif self.JK == 'last':
            node_embedding = h_list[-1]  # (num_nodes, d_emb)
        elif self.JK == 'max':
            h_list = [h.unsqueeze_(0) for h in h_list]  # dim 0: layer index
            # torch.max() -> (value, index) for each feat dim of each node
            node_embedding = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == 'sum':
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_embedding = torch.sum(torch.cat(h_list, dim=0), dim=0)
        
        return node_embedding

class GraphEncoder(nn.Module):
    def __init__(self, num_layers, d_emb, JK='last', dropout=0., gnn_type='gin', graph_pooling='mean', num_heads=1):
        super().__init__()
        if num_layers < 2:
            raise ValueError('GNN layers must > 1.')
        
        self.node_encoder = NodeEncoder(num_layers, d_emb, JK, dropout, gnn_type, num_heads)

        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'attention':
            if JK == 'concat':
                self.pool = GlobalAttention(gate_nn=nn.Linear((num_layers + 1) * d_emb, 1))
            else:
                self.pool = GlobalAttention(gate_nn=nn.Linear(d_emb, 1))
        else:
            raise ValueError(f"Unsupported graph pooling type: {graph_pooling}")
    
    def forward(self, *args):
        if len(args) == 4:
            x, edge_index, edge_attr, batch = args[0], args[1], args[2], args[3]
        elif len(args) == 1:
            gdata = args[0]
            x, edge_index, edge_attr, batch = gdata.x, gdata.edge_index, gdata.edge_attr, gdata.batch
        else:
            raise ValueError('Invalid input arguments.')
            
        node_embedding = self.node_encoder(x, edge_index, edge_attr)
        graph_embedding = self.pool(node_embedding, batch)  # (batch_size, d_emb)
                
        return graph_embedding

if __name__ == '__main__':
    pass
