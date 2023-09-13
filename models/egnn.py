import jax 
import jax.numpy as jnp
import jax.random as random

import haiku as hk

from einops import rearrange

from models.model import Model 

def gen_edges (batch_size, n_nodes):
    r = jnp.linspace(0, n_nodes-1, n_nodes).repeat(n_nodes).astype(int)
    r = jnp.expand_dims(r, 0).repeat(batch_size, 0)
    c = jnp.linspace(0, n_nodes-1, n_nodes).reshape((1, n_nodes)).repeat(n_nodes, axis=0)
    c = c.flatten().astype(int)
    c = jnp.expand_dims(c, 0).repeat(batch_size, 0)
    return (r,c)


def coord2radial(edge_index, coord, normalize=True, epsilon=1e-8):
    row, col = edge_index
    diff = coord[row] - coord[col]
    rad = (diff**2).sum(-1)
    rad = rearrange(rad, 'i -> i ()')

    if normalize:
        norm= jnp.sqrt(jax.lax.stop_gradient(rad)) + epsilon
        diff = diff / norm

    return rad, diff

class EGCL(hk.Module):
    """
    E(n) Equivariant Graph Convolutional layers. Taken from: https://arxiv.org/pdf/2102.09844.pdf
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
        self.coord_op = hk.Sequential([
            hk.Linear(hidden_nf),
            activation,
            hk.Linear(1)
        ])
        self.out_nf = out_nf
        self.hidden_nf = hidden_nf
        self.activation = activation
        self.normalize = normalize

    def __call__(self, h, coord, edge_index, edge_attr=None, node_attr=None):
        r, c = edge_index
        rad, diff = coord2radial(edge_index, coord)

        if edge_attr is not None:
            edge_input = jnp.concatenate([h[r], h[c], rad, edge_attr], axis=1)
        else:
            edge_input = jnp.concatenate([h[r], h[c], rad], axis=1)
        m_ij = self.edge_op(edge_input)

        weights = self.coord_op(m_ij)
        coors_sum = jax.ops.segment_sum(diff * weights, r, num_segments=coord.shape[0])
        coord = coord + coors_sum

        m_i = jax.ops.segment_sum(m_ij, r, num_segments=h.shape[0])
        if node_attr is not None:
            agg = jnp.concatenate([h, m_i, node_attr], 1)
        else:
            agg = jnp.concatenate([h, m_i], 1)
        h_out = h + self.node_op(agg)

        return h_out, coord

class EGNN_Module(hk.Module):
    def __init__(
        self,
        batch_size,
        n_nodes,
        hidden_nf,
        out_nf,
        n_layers,
        activation,
        reg,
        normalize=True,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.reg = reg

        self.embedding_in = hk.Sequential([hk.Linear(hidden_nf), activation, hk.Linear(hidden_nf)])

        self.embedding_out = hk.Sequential([hk.Linear(hidden_nf), activation, hk.Linear(out_nf)])
        
        self.edges = gen_edges(batch_size, n_nodes)

        self.layers = []
        for _ in range(n_layers):
            self.layers.append(EGCL(hidden_nf, hidden_nf, activation, normalize))

    def __call__(self, inputs, key):
        h = inputs.h; x = inputs.x; edges = self.edges; edge_attr = inputs.edge_attr
        h = self.embedding_in(h)

        for i in range(self.n_layers):
            h, x = jax.vmap(self.layers[i])(h, x, edges, edge_attr)
            x -= x*  self.reg

        h = rearrange(h, 'b n d -> b (n d)')
        
        return self.embedding_out(h)

def EGNN(hidden_nf, out_nf, n_layers, activation, normalize=True):
    def egnn(inputs, key):
        batch_size = inputs.x.shape[0]
        n_nodes = inputs.x.shape[1]
        module = EGNN_Module(batch_size, n_nodes, hidden_nf, out_nf, n_layers, activation, normalize)
        return module(inputs, key)
    
    return Model(egnn)