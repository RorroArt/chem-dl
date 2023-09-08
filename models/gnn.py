import jax
import jax.numpy as jnp

import haiku as hk

from einops import rearrange

from models.model import Model
class GCL(hk.Module):
    """
    Graph Convolutional layer. Similar to the one in https://arxiv.org/pdf/1704.01212.pdf.
    """
    def __init__(
        self,
        out_nf,
        hidden_nf,
        activation,
        normalize=True,
    ):
        super().__init__()
        self.edge_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf),
            activation
        ])

        self.node_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(out_nf)
        ])

        
        self.out_nf = out_nf
        self.hidden_nf = hidden_nf
        self.activation = activation

    def __call__(self, h, edge_index, coord, edge_attr=None):
        r = edge_index[:, 0]; c = edge_index[:, 1]
        print(r.shape)
        if edge_attr is not None:
            edge_input = jnp.concatenate([h[r], h[ c], edge_attr], axis=1)
        else:
            edge_input = jnp.concatenate([h[r], h[c]], axis=1)

        m_ij = self.edge_op(edge_input)


        m_i = jax.ops.segment_sum(m_ij, r, num_segments=h.shape[0])
        

        agg = jnp.concatenate([h, m_i], 1)

        h_out = h + self.node_op(agg)

        return h_out, coord
    
class GNN_Module(hk.Module):
    def __init__(
        self,
        hidden_nf,
        out_nf,
        n_layers,
        activation,
        normalize=True,
    ):
        super().__init__()
        self.n_layers = n_layers

        self.embedding_in = hk.Sequential([hk.Linear(hidden_nf), activation, hk.Linear(hidden_nf)])

        self.embedding_out = hk.Sequential([hk.Linear(hidden_nf), activation, hk.Linear(out_nf)])
        
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(GCL(hidden_nf, hidden_nf, activation, normalize))

    def __call__(self, inputs, key):
        h, edges, edge_attr = inputs.x, inputs.edges, inputs.edge_attr
        h = self.embedding_in(h)

        for i in range(self.n_layers):
            h = self.layers[i](h, edges, edge_attr)

        h = rearrange(h, 'n d -> (n d)')
        
        return self.embedding_out(h)

def GNN(hidden_nf, out_nf, n_layers, activation, normalize=True):
    def gnn(inputs, key):
        module = GNN_Module(hidden_nf, out_nf, n_layers, activation, normalize)
        return module(inputs, key)
    
    return Model(gnn)