import torch
import torch.nn.functional as F
from torch.nn import Linear, GRUCell
from torch_geometric.nn import aggr, GATv2Conv, TransformerConv, SAGPooling, MLP

from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
from torch_geometric.typing import Metadata, NodeType, EdgeType

class AttentiveGnLConv(torch.nn.Module):
    def __init__(self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_heads: Optional[int] = 4,
        edge_dim: Optional[int] = -1,
        dropout: Optional[float] = 0.,
        conv_kwargs: Optional[dict] = {},
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.dropout = dropout

        if num_heads == 1:
            self.conv1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim,
                                   dropout=dropout, **conv_kwargs)
            self.gru1 = GRUCell(hidden_channels, in_channels)
            self.lin1 = Linear(in_channels, hidden_channels)
        elif num_heads >= 2:
            self.conv1 = TransformerConv(in_channels, hidden_channels//2, edge_dim=edge_dim,
                                   heads=num_heads, dropout=dropout, **conv_kwargs)
            self.lin0 = Linear(in_channels, hidden_channels*(num_heads//2))
            self.gru1 = GRUCell((hidden_channels//2)*num_heads, hidden_channels*(num_heads//2))
            self.lin1 = Linear(hidden_channels*(num_heads//2), hidden_channels)

        self.convs = torch.nn.ModuleList()
        self.grus = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim,
                                        add_self_loops=False, dropout=dropout, **conv_kwargs))
            self.grus.append(GRUCell(hidden_channels, hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.gru1.reset_parameters()
        self.lin0.reset_parameters() if hasattr(self, 'lin0') else None
        self.lin1.reset_parameters()
        for conv, gru in zip(self.convs, self.grus):
            conv.reset_parameters()
            gru.reset_parameters()

    def forward(self, 
                x: Tensor,
                edge_index: Tensor, 
                edge_attr: Tensor
            ) -> Tensor:
        # Atom Embedding:
        if self.num_heads == 1:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        elif self.num_heads >= 2:
            h = F.elu_(self.conv1(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.leaky_relu_(self.lin0(x))
            x = self.gru1(h, x).relu_()
            x = F.leaky_relu_(self.lin1(x))
        g = [x]

        for conv, gru in zip(self.convs, self.grus):
            h = F.elu(conv(x, edge_index, edge_attr))
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x).relu()
            g.append(x)

        return sum(g)   # sum hierarchical node embeddings

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'in_channels={self.in_channels}, '
                f'hidden_channels={self.hidden_channels}, '
                f'num_layers={self.num_layers}, '
                f'num_heads={self.num_heads}, '
                f'dropout={self.dropout})'
                f')')

class GnLTransformer_Paired(torch.nn.Module):
    def __init__(self,
        dim_in_G: int,
        dim_in_L: int,
        dim_h_conv: int,
        dim_h_lin: int,
        dim_out: int,
        num_layer_conv: int, 
        num_layer_lin: int,
        num_heads: Optional[int] = 4,
        pool_k_G: Optional[int] = 20,
        pool_k_L: Optional[int] = 20,
        dropout: Optional[float] = 0.,
    ):
        super().__init__()
        # torch.manual_seed(42)
        self.conv_G = AttentiveGnLConv(in_channels=dim_in_G,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout)
        self.conv_L = AttentiveGnLConv(in_channels=dim_in_L,
                                hidden_channels=dim_h_conv,
                                num_layers=num_layer_conv,
                                num_heads=num_heads,
                                dropout=dropout)

        self.pool_G = SAGPooling(dim_h_conv, ratio=pool_k_G)#, GNN=GATv2Conv)
        self.pool_L = SAGPooling(dim_h_conv, ratio=pool_k_L)#, GNN=GATv2Conv)
        self.sort_G = aggr.SortAggregation(k=pool_k_G)
        self.sort_L = aggr.SortAggregation(k=pool_k_L)

        self.mlp = MLP(in_channels=dim_h_conv*(pool_k_G+pool_k_L),
                       hidden_channels=dim_h_lin,
                       out_channels=dim_out,
                       num_layers=num_layer_lin,
                       dropout=dropout)

        self.dim_h_conv = dim_h_conv
        self.dim_h_lin = dim_h_lin
        self.dim_out = dim_out
        self.num_layer_conv = num_layer_conv
        self.num_layer_lin = num_layer_lin
        self.num_heads = num_heads
        self.pool_k_G = pool_k_G
        self.pool_k_L = pool_k_L
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        self.conv_G.reset_parameters()
        self.conv_L.reset_parameters()
        self.pool_G.reset_parameters()
        self.pool_L.reset_parameters()
        self.mlp.reset_parameters()

    def forward(self, data_G, data_L):
        x_G, edge_index_G, edge_attr_G, batch_G = data_G.x, data_G.edge_index, data_G.edge_attr, data_G.batch
        x_L, edge_index_L, edge_attr_L, batch_L = data_L.x, data_L.edge_index, data_L.edge_attr, data_L.batch
        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return x

class GnLTransformer_Hetero(GnLTransformer_Paired):
    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        x_G, edge_index_G, edge_attr_G, batch_G = x_dict['node'], edge_index_dict[('node', 'n2n', 'node')], edge_attr_dict[('node', 'n2n', 'node')], batch_dict['node']
        x_L, edge_index_L, edge_attr_L, batch_L = x_dict['edge'], edge_index_dict[('edge', 'e2e', 'edge')], edge_attr_dict[('edge', 'e2e', 'edge')], batch_dict['edge']

        x_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L = self.conv_L(x_L, edge_index_L, edge_attr_L)

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)

        return x

# X... classes for visualization and explanation
class XAGnLConv(AttentiveGnLConv):
    def forward(self, x, edge_index, edge_attr):
        vis_data = {}
        if self.num_heads == 1:
            h, att = self.conv1(x, edge_index, edge_attr, 
                                return_attention_weights=True)
            # vis_data['att_eid_conv1'] = att[0].detach().cpu().numpy()
            vis_data['att_w_conv1'] = att[1].detach().cpu().numpy()
            h = F.elu_(h)
            vis_data['x_conv1'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = self.gru1(h, x).relu_()
            x = self.lin1(x)
            x = F.leaky_relu_(x)
            vis_data['x_gru1'] = x.clone().squeeze().detach().cpu().numpy()
        elif self.num_heads >= 2:
            h, att = self.conv1(x, edge_index, edge_attr,
                                return_attention_weights=True)
            # vis_data['att_eid_conv1'] = att[0].detach().cpu().numpy()
            vis_data['att_w_conv1'] = att[1].detach().cpu().numpy()
            h = F.elu_(h)
            vis_data['x_conv1'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = F.leaky_relu_(self.lin0(x))
            x = self.gru1(h, x).relu_()
            x = self.lin1(x)
            x = F.leaky_relu_(x)
            vis_data['x_gru1'] = x.clone().squeeze().detach().cpu().numpy()
        g = [x]

        for i, (conv, gru) in enumerate(zip(self.convs, self.grus)):
            h, att = conv(x, edge_index, edge_attr, 
                          return_attention_weights=True)
            # vis_data[f'att_eid_conv{i+2}'] = att[0].detach().cpu().numpy()
            vis_data[f'att_w_conv{i+2}'] = att[1].detach().cpu().numpy()
            h = F.elu(h)
            vis_data[f'x_conv{i+2}'] = h.clone().squeeze().detach().cpu().numpy()
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = gru(h, x)
            x = F.relu(x)
            vis_data[f'x_gru{i+2}'] = x.clone().squeeze().detach().cpu().numpy()
            g.append(x)
            # sum hierarchical node embeddings
        g = sum(g)
        return g, vis_data

class XGnLTransformer_Paired(GnLTransformer_Paired):
    def __init__(self, dim_in_G, dim_in_L, dim_h_conv, dim_h_lin, dim_out, 
            num_layer_conv, num_layer_lin, num_heads, pool_k_G, pool_k_L, dropout=0.):
        super().__init__(dim_in_G, dim_in_L, dim_h_conv, dim_h_lin, dim_out, 
                num_layer_conv, num_layer_lin, num_heads, pool_k_G, pool_k_L, dropout)
        self.conv_G = XAGnLConv(dim_in_G, dim_h_conv, num_layer_conv, num_heads, dropout)
        self.conv_L = XAGnLConv(dim_in_L, dim_h_conv, num_layer_conv, num_heads, dropout)
    def forward(self, data_G, data_L):
        vis_data = {}
        x_G, edge_index_G, edge_attr_G, batch_G = data_G.x, data_G.edge_index, data_G.edge_attr, data_G.batch
        x_L, edge_index_L, edge_attr_L, batch_L = data_L.x, data_L.edge_index, data_L.edge_attr, data_L.batch
        x_G, vis_data_G = self.conv_G(x_G, edge_index_G, edge_attr_G)
        x_L, vis_data_L = self.conv_L(x_L, edge_index_L, edge_attr_L)
        vis_data['G_node'] = vis_data_G
        vis_data['L_node'] = vis_data_L

        x_G, _, _, batch_G, _, _ = self.pool_G(x_G, edge_index_G, edge_attr_G, batch_G)
        x_L, _, _, batch_L, _, _ = self.pool_L(x_L, edge_index_L, edge_attr_L, batch_L)

        x_G = self.sort_G(x_G, batch_G)
        x_L = self.sort_L(x_L, batch_L)
        vis_data['G_graph1'] = x_G.clone().squeeze().detach().cpu().numpy()
        vis_data['L_graph1'] = x_L.clone().squeeze().detach().cpu().numpy()

        x = torch.cat([x_G, x_L], dim=1)
        x = self.mlp(x)
        vis_data['GnL_graph-1'] = x.clone().squeeze().detach().cpu().numpy()
        return x, vis_data