import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ..poly2graph import spectral_potential, PosGoL
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def normalize_color(color, vmin=.3):
    color -= np.min(color)
    ptp = color.max()
    if ptp == 0: return np.full_like(color, vmin)
    else: return color/ptp*(1-vmin) + vmin

def visualize_attention_scores(G, pos, labels, node_att, edge_att, is_G, ax):
    node_options = {
        'node_size': 200*node_att,
        'node_color': '#A60628' if is_G else '#348ABD',
    }
    label_options = {
        'labels': labels,
        'font_color': 'w',
        'font_weight': 'bold',
        # 'font_family': 'Georgia',
        'font_size': 6.2
    }
    edge_options = {
        'edge_color': edge_att,
        'edge_cmap': plt.cm.Blues if is_G else plt.cm.Greens,
        'edge_vmin': 0.,
        'width': 3*edge_att,
        'connectionstyle': "arc3,rad=0.08" if G.__class__ == nx.DiGraph else "arc3"
    }
    nx.draw_networkx_nodes(G, pos=pos, ax=ax, **node_options)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, **label_options)
    nx.draw_networkx_edges(G, pos=pos, ax=ax, **edge_options)
    ax.axis('off')
    return ax

def visualize_node_embeddings(x, sorted_idx, is_G, ax):
    # similarity = np.corrcoef(x[sorted_idx])
    similarity = cosine_similarity(x[sorted_idx])
    cmap = 'inferno' if is_G else 'viridis'
    if is_G: tick_labels = sorted_idx
    else: tick_labels = [chr(65+i) for i in sorted_idx]
    ax.imshow(similarity, cmap=cmap)
    ticks = np.arange(0, len(sorted_idx), 1)
    ax.set_xticks(ticks, labels=tick_labels)
    ax.set_yticks(ticks, labels=tick_labels)
    ax.tick_params(axis='both', which='both', direction='in')
    return ax

class ExplanationSummary():
    def __init__(self, model, dataset, graph_class_index=None, idx=None, y_true=None):
        self.model = model.cpu()
        self.dataset = dataset
        if graph_class_index is not None:
            assert len(dataset) == len(graph_class_index), 'graph_class_index must have the same length as the dataset.'
        self.graph_class_index = graph_class_index
        self.embeddings = None
        if idx is not None:
            self.idx = idx
            self.get_data(idx, y_true)
            self.to_networkx_Graph()
    
    def get_data(self, idx, y_true=None):
        data_G, data_L = self.dataset[idx]
        data_G.batch = torch.zeros(data_G.num_nodes, dtype=torch.long)
        data_L.batch = torch.zeros(data_L.num_nodes, dtype=torch.long)
        self.model.eval()
        out, embeddings = self.model(data_G, data_L)
        probs = F.softmax(out, dim=1).squeeze().detach().numpy()
        y_pred = probs.argsort()[::-1]
        self.y_pred = y_pred
        self.y_pred_probs = probs[y_pred]
        self.y_true = data_G.y.item() if y_true is None else y_true
        self.poly_coeffs = data_G.full_coeffs.numpy()[0]
        if self.graph_class_index is not None:
            self.graph_iso_class = self.graph_class_index[idx].item()
        self.embeddings = embeddings
        self.pygG = data_G
        self.pygL = data_L

    def __call__(self, idx, y_true=None):
        self.idx = idx
        self.get_data(idx, y_true)
        self.to_networkx_Graph()
        return self

    def clear_embeddings(self):
        self.embeddings = None

    def _get_edge_attention_TransformerConv(self, head):
        '''head is the index of the attention head, starting from 1'''
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        edge_att_G = self.embeddings['G_node']['att_w_conv1']
        edge_att_L = self.embeddings['L_node']['att_w_conv1']
        if head == 'all':
            return edge_att_G.sum(-1), edge_att_L.sum(-1)
        elif isinstance(head, int):
            return edge_att_G[:,head-1], edge_att_L[:,head-1]
        else:
            head = np.asarray(head)-1
            return edge_att_G[:,head].sum(-1), edge_att_L[:,head].sum(-1)

    def _get_edge_attention_GATv2Conv(self, layer):
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        if isinstance(layer, int):
            if layer > self.model.num_layer_conv or layer < 2:
                raise ValueError(f'Layer must be an int between 2 and {self.model.num_layer_conv}.')
            edge_att_G = self.embeddings['G_node'][f'att_w_conv{layer}']
            edge_att_L = self.embeddings['L_node'][f'att_w_conv{layer}']
            return edge_att_G.ravel(), edge_att_L.ravel()
        elif layer == 'all':
            layer = range(2, self.model.num_layer_conv+1)
        edge_att_G = np.hstack([self.embeddings['G_node'][f'att_w_conv{i}'] for i in layer])
        edge_att_L = np.hstack([self.embeddings['L_node'][f'att_w_conv{i}'] for i in layer])
        return edge_att_G.sum(-1), edge_att_L.sum(-1)

    def _get_node_attention(self,
        target_node_index_G, edge_att_G,
        target_node_index_L, edge_att_L,
        aggr=np.mean):
        '''Node attention is the average of incoming edge attention scores'''
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        node_att_G = [aggr(edge_att_G[target_node_index_G==i]) for i in range(self.pygG.num_nodes)]
        node_att_L = [aggr(edge_att_L[target_node_index_L==i]) for i in range(self.pygL.num_nodes)]
        return normalize_color(node_att_G), normalize_color(node_att_L)

    def get_node_embeddings(self, is_G=True, idx=None):
        if idx is not None: self(idx)
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        if is_G or is_G == 'G':
            emb = [self.pygG.x.numpy()]
            emb_labels = ['Input, G']
            for i in range(1, self.model.num_layer_conv+1):
                emb.append(self.embeddings['G_node'][f'x_conv{i}'])
                emb.append(self.embeddings['G_node'][f'x_gru{i}'])
                emb_labels.extend([f'Conv, Layer {i}, G', f'GRU, Layer {i}, G'])
            return emb, emb_labels
        elif not is_G or is_G == 'L':
            emb = [self.pygL.x.numpy()]
            emb_labels = ['Input, L']
            for i in range(1, self.model.num_layer_conv+1):
                emb.append(self.embeddings['L_node'][f'x_conv{i}'])
                emb.append(self.embeddings['L_node'][f'x_gru{i}'])
                emb_labels.extend([f'Conv, Layer {i}, L', f'GRU, Layer {i}, L'])
            return emb, emb_labels
        else:
            raise ValueError('is_G must be a boolean or a string of "G" or "L".')

    
    def get_graph_embeddings(self, idx=None):
        if idx is not None: self(idx)
        if self.embeddings is None:
            raise ValueError('Embeddings not found. Call get_data() before calling this method.')
        emb_convG = self.embeddings['G_graph1']
        emb_convL = self.embeddings['L_graph1']
        emb_last = self.embeddings['GnL_graph-1']
        return emb_convG, emb_convL, emb_last
    
    def to_networkx_Graph(self, head='all', layer='all', create_using=nx.Graph):
        # get edge index
        edge_index_G = self.pygG.edge_index.numpy()
        edge_index_L = self.pygL.edge_index.numpy()

        # get edge attention scores
        edge_att_G, edge_att_L = [], []
        if head is not None:
            att_G, att_L = self._get_edge_attention_TransformerConv(head)
            edge_att_G.append(att_G)
            edge_att_L.append(att_L)
        if layer is not None:
            att_G, att_L = self._get_edge_attention_GATv2Conv(layer)
            edge_att_G.append(att_G)
            edge_att_L.append(att_L)
        edge_att_G = np.sum(edge_att_G, axis=0)
        edge_att_L = np.sum(edge_att_L, axis=0)

        # get node attention scores
        node_att_G, node_att_L = self._get_node_attention(
                                    edge_index_G[1], edge_att_G,
                                    edge_index_L[1], edge_att_L)
        
        # Create networkx graphs
        nxG = create_using(); nxL = create_using()
        # Add nodes with positions and colors
        for i in range(self.pygG.num_nodes):
            nxG.add_node(i, pos=self.pygG.pos[i], att=node_att_G[i], label=i)
        for i in range(self.pygL.num_nodes):
            # alphabetic labels, lowercase
            nxL.add_node(i, pos=self.pygL.pos[i], att=node_att_L[i], label=chr(65+i))
        # Add edges with positions and colors
        for i, (s, e) in enumerate(edge_index_G.T):
            if nxG.has_edge(s, e):
                nxG[s][e]['att'] += edge_att_G[i]
            else:
                nxG.add_edge(s, e, att=edge_att_G[i])
        for i, (s, e) in enumerate(edge_index_L.T):
            if nxL.has_edge(s, e):
                nxL[s][e]['att'] += edge_att_L[i]
            else:
                nxL.add_edge(s, e, att=edge_att_L[i])
        # Normalize edge attributes for edge color
        edge_color_G = normalize_color(list(nx.get_edge_attributes(nxG, 'att').values()))
        edge_color_L = normalize_color(list(nx.get_edge_attributes(nxL, 'att').values()))
        # Set the normalized edge colors as attributes
        nx.set_edge_attributes(nxG, dict(zip(nxG.edges, edge_color_G)), 'att')
        nx.set_edge_attributes(nxL, dict(zip(nxL.edges, edge_color_L)), 'att')

        self.nxG = nxG
        self.nxL = nxL
        self.edge_color_G = edge_color_G
        self.edge_color_L = edge_color_L
        self.node_color_G = node_att_G
        self.node_color_L = node_att_L

        return nxG, nxL


    def attention_summary(self, ax=None):
        '''
        Plot 0: DOS, 1: attention scores of G-channel, 2: attention scores of L-channel
        ax: 1x3 array of matplotlib axes
        '''
        if ax is None:
            fig, ax = plt.subplots(1, 3, figsize=(9,3))
        emax = self.pygG.Emax.numpy()[0]

        # plot 0: DOS
        img = PosGoL(spectral_potential(
            c=self.poly_coeffs,
            Emax=emax,
            Elen=200
        ), ksizes=[11])
        ax[0].imshow(img, cmap='gray', extent=emax, aspect='equal')
        ax[0].set_title('Density of States')
        ax[0].set_xlabel('Re(E)', labelpad=.01)
        ax[0].set_ylabel('Im(E)', labelpad=.01)
        ax[0].tick_params(axis='both', which='both', direction='in', color='w', pad=2)
        ax[0].tick_params(axis='y', which='both', labelrotation=90)
        
        # plot 1: attention scores of G-channel
        self.to_networkx_Graph(head='all', layer='all')
        visualize_attention_scores(
            self.nxG,
            self.pygG.pos,
            {i: f'{i}' for i in range(self.pygG.num_nodes)},
            self.node_color_G,
            self.edge_color_G,
            is_G=True, ax=ax[1]
        )
        ax[1].set(
            title='Attention Weights of G',
        )

        # plot 2: attention scores of L-channel
        visualize_attention_scores(
            self.nxL,
            self.pygL.pos,
            {i: f'{chr(65+i)}' for i in range(self.pygL.num_nodes)},
            self.node_color_L,
            self.edge_color_L,
            is_G=False, ax=ax[2]
        )
        ax[2].set(
            title='Attention Weights of L',
        )

        return ax
    
    def node_embedding_summary(self, ax=None):
        '''
        Plot 0-2: node embeddings of G-channel, 3-5: node embeddings of L-channel
        ax: 1x6 array of matplotlib axes
        '''
        if ax is None:
            fig, axes = plt.subplots(2, 3, figsize=(9,6))
            ax = axes.flat

        # plot 0-2: node embeddings of G-channel
        emb_G, emb_labels_G = self.get_node_embeddings(is_G=True)
        sorted_idx_G = PCA(n_components=1).fit_transform(emb_G[-1]).ravel().argsort()
        for i, layer in enumerate([0,2,-1]):
            visualize_node_embeddings(emb_G[layer], sorted_idx_G, is_G=True, ax=ax[i])
            ax[i].set(title=emb_labels_G[layer])

        # plot 3-5: node embeddings of L-channel
        emb_L, emb_labels_L = self.get_node_embeddings(is_G=False)
        sorted_idx_L = PCA(n_components=1).fit_transform(emb_L[-1]).ravel().argsort()
        for i, layer in enumerate([0,2,-1]):
            visualize_node_embeddings(emb_L[layer], sorted_idx_L, is_G=False, ax=ax[i+3])
            ax[i+3].set(title=emb_labels_L[layer])

        return ax

    def summary_plot(self, path=None):
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        ax = axes.flat

        # plot 0-2: DOS, attention scores
        self.attention_summary(ax[:3])

        # plot 3-8: node embeddings
        self.node_embedding_summary(ax[3:])

        plt.tight_layout(pad=.01, h_pad=.01, w_pad=.01)
        if path is not None:
            plt.savefig(path, tranparent=True)

        return fig, axes
    
    def summary_plot_per_layer(self, path=None):
        n_row = 1+self.model.num_layer_conv
        n_col = max(self.model.num_heads+2, 2*3)
        fig, axes = plt.subplots(n_row, n_col, figsize=(3*n_col, 3*n_row))

        emb_G, emb_labels_G = self.get_node_embeddings(is_G=True)
        sorted_idx_G = PCA(n_components=1).fit_transform(emb_G[-1]).ravel().argsort()
        emb_L, emb_labels_L = self.get_node_embeddings(is_G=False)
        sorted_idx_L = PCA(n_components=1).fit_transform(emb_L[-1]).ravel().argsort()

        # each head, TransformerConv
        for col in range(self.model.num_heads):
            self.to_networkx_Graph(head=col+1, layer=None)
            visualize_attention_scores(self.nxG, self.pygG.pos,
                {i: f'{i}' for i in range(self.pygG.num_nodes)},
                self.node_color_G, self.edge_color_G,
                is_G=True, ax=axes[0, col])
            axes[0, col].set(title=f'Transformer, Layer 1, Head {col+1}')
            visualize_attention_scores(self.nxL, self.pygL.pos,
                {i: f'{chr(65+i)}' for i in range(self.pygL.num_nodes)},
                self.node_color_L, self.edge_color_L,
                is_G=False, ax=axes[1, col])
            axes[1, col].set(title=f'Transformer, Layer 1, Head {col+1}')
        
        # each layer > 1, GATv2Conv
        for row in range(2, n_row):
            self.to_networkx_Graph(head=None, layer=row)
            visualize_attention_scores(self.nxG, self.pygG.pos,
                {i: f'{i}' for i in range(self.pygG.num_nodes)},
                self.node_color_G, self.edge_color_G,
                is_G=True, ax=axes[row, 0])
            axes[row, 0].set(title=f'GATv2, Layer {row}')
            visualize_attention_scores(self.nxL, self.pygL.pos,
                {i: f'{chr(65+i)}' for i in range(self.pygL.num_nodes)},
                self.node_color_L, self.edge_color_L,
                is_G=False, ax=axes[row, 3])
            axes[row, 3].set(title=f'GATv2, Layer {row}')
        
        # node embeddings, 1st layer
        for i in range(1, 3):
            visualize_node_embeddings(emb_G[i], sorted_idx_G, is_G=True, ax=axes[0, -i])
            axes[0, -i].set(title=emb_labels_G[i])
            visualize_node_embeddings(emb_L[i], sorted_idx_L, is_G=False, ax=axes[1, -i])
            axes[1, -i].set(title=emb_labels_L[i])
            
        # node embeddings, >1st layer
        for row in range(2, n_row):
            ax1 = visualize_node_embeddings(emb_G[2*row-1], sorted_idx_G, is_G=True, ax=axes[row, 1])
            ax2 = visualize_node_embeddings(emb_G[2*row], sorted_idx_G, is_G=True, ax=axes[row, 2])
            ax1.set(title=emb_labels_G[2*row-1])
            ax2.set(title=emb_labels_G[2*row])
            ax4 = visualize_node_embeddings(emb_L[2*row-1], sorted_idx_L, is_G=False, ax=axes[row, 4])
            ax5 = visualize_node_embeddings(emb_L[2*row], sorted_idx_L, is_G=False, ax=axes[row, 5])
            ax4.set(title=emb_labels_L[2*row-1])
            ax5.set(title=emb_labels_L[2*row])

        plt.tight_layout(pad=.01, h_pad=.01, w_pad=.01)
        if path is not None:
            plt.savefig(path, tranparent=True)

        return fig, axes