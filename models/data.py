import jax
import jax.numpy as jnp

from typing import NamedTuple, Optional

class Batch(NamedTuple):
    x: jnp.ndarray
    edges: Optional[jnp.ndarray]
    edge_attr: Optional[jnp.ndarray]