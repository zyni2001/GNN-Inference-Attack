import torch
import torch.nn.functional as F
from torch.nn import Linear

from lib_gnn_model.diffpool.diffpool_net import GNN
from torch_geometric.utils import dense_to_sparse

from functools import partial
from torch.nn import Linear, BatchNorm1d, Sequential, ReLU
from torch_geometric.nn import global_mean_pool, global_add_pool, GINConv, GATConv
from ..gcn_conv import GCNConv
from torch_geometric.nn import global_mean_pool

# class MeanPoolNet(torch.nn.Module):
#     def __init__(self, num_feats, num_classes, hidden_channels=64):
#         super(MeanPoolNet, self).__init__()
#         self.conv = GNN(num_feats, hidden_channels, hidden_channels, lin=False)
        
#         self.lin1 = Linear(3 * hidden_channels, hidden_channels)
#         self.lin2 = Linear(hidden_channels, num_classes)

#     def forward(self, x, adj, mask=None):
#         x = F.relu(self.conv(x, adj, mask))
        
#         self.graph_embedding = x.mean(dim=1)
        
#         x = F.relu(self.lin1(self.graph_embedding))
#         x = self.lin2(x)
        
#         return F.log_softmax(x, dim=-1)

class MeanPoolNet(torch.nn.Module):
    """GCN with BN and residual connection."""
    def __init__(self, num_feats, num_classes, num_feat_layers=1, num_conv_layers=3,
                 num_fc_layers=2, gfn=False, collapse=False, residual=False,
                 res_branch="BNConvReLU", global_pool="sum", dropout=0, 
                 edge_norm=True):
        super(MeanPoolNet, self).__init__()

        self.global_pool = global_mean_pool
        self.dropout = dropout
        GConv = partial(GCNConv, edge_norm=edge_norm, gfn=gfn)

        hidden_in = num_feats
        hidden = 192
        self.bn_feat = BatchNorm1d(hidden_in)
        self.conv_feat = GCNConv(hidden_in, hidden, gfn=True) # linear transform
        self.bns_conv = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()

        for i in range(num_conv_layers):
            self.bns_conv.append(BatchNorm1d(hidden))
            self.convs.append(GConv(hidden, hidden))
        self.bn_hidden = BatchNorm1d(hidden)
        self.bns_fc = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()

        for i in range(num_fc_layers - 1):
            self.bns_fc.append(BatchNorm1d(hidden))
            self.lins.append(Linear(hidden, hidden))
        self.lin_class = Linear(hidden, num_classes)

        # BN initialization.
        for m in self.modules():
            if isinstance(m, (torch.nn.BatchNorm1d)):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0.0001)

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, _ = x.shape
        edge_indices = [dense_to_sparse(adj[i])[0] for i in range(batch_size)]
        
        # Concatenate edge indices and add offsets
        edge_index = torch.cat([edge_indices[i] + i * num_nodes for i in range(batch_size)], dim=1)
        
        # Flatten x and mask for processing
        x = x.view(-1, x.size(-1))
        if mask is not None:
            mask = mask.view(-1)
        
        # Create batch vector
        batch = torch.arange(batch_size, device=x.device).view(-1, 1).repeat(1, num_nodes).view(-1)
        
        x = self.bn_feat(x)
        x = F.relu(self.conv_feat(x, edge_index))
        
        for i, conv in enumerate(self.convs):
            x = self.bns_conv[i](x)
            x = F.relu(conv(x, edge_index, edge_weight=None))  # Not using mask as edge weights

        x = self.global_pool(x, batch)
        for i, lin in enumerate(self.lins):
            x = self.bns_fc[i](x)
            x = F.relu(lin(x))

        self.graph_embedding = x

        x = self.bn_hidden(x)
        if self.dropout > 0:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_class(x)
        return F.log_softmax(x, dim=-1)
