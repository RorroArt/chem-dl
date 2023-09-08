import jax 
import jax.numpy as jnp

from models import Model
from models.gnn import GNN 
from models.data import Batch

model = GNN(
    hidden_nf=128, 
    out_nf=4, 
    n_layers=2, 
    activation=jax.nn.swish,  
    normalize=True
)


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    node_features = jnp.ones(( 5, 3)) # 5 nodes, 3 channels 
    edges = jnp.array([[0,1],[1,3], [1,4], [2,3], [2,5]]).astype(jnp.int32) # Edge list
    edge_attr = jnp.ones((5, 2)) # 5 edges, 1 channel
    
    inputs = Batch(x=node_features, edges=edges, edge_attr=edge_attr)
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)