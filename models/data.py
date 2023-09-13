import jax
import jax.numpy as jnp

from jax.random import permutation, PRNGKey

from typing import NamedTuple, Optional

class Batch(NamedTuple):
    x: jnp.ndarray
    h: Optional[jnp.ndarray] = None
    edge_attr: Optional[jnp.ndarray] = None
    properties: Optional[jnp.ndarray] = None

class DataLoader:
    __slots__ = 'indeces', 'coords', 'node_features', 'edges', 'edge_attr', 'batch_size', 'idx', 'seed'

    def __init__(self, indeces, coords, node_features = None, edges=None, edge_attr=None, batch_size=1):
        self.indeces = indeces
        self.coords = coords
        self.node_features = node_features
        self.edges = edges
        self.edge_attr = edge_attr
        self.batch_size = batch_size
        self.idx = 0
        self.seed = 1
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.idx < len(self.indeces) - self.batch_size:
            self.idx += self.batch_size
            batch_idx = self.indeces[self.idx - self.batch_size: self.idx]

            batch = Batch(
                x=self.coords[batch_idx],
                h=self.node_features[batch_idx] if self.node_features is not None else None,
                edges=self.edges[batch_idx] if self.edges is not None else None,
                edge_attr=self.edge_attr[batch_idx] if self.edge_attr is not None else None,
            )   
            return batch
        else:
            self.indeces = permutation(PRNGKey(self.seed), self.indeces)
            self.idx = 0
            self.seed += 1
            raise StopIteration 
    
    def __len__(self):
        return len(self.indeces)