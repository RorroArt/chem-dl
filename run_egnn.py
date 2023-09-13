import jax 
import jax.numpy as jnp

from models.egnn import EGNN 
from models.data import Batch

N_NODES = 5
BATCH_SIZE =10

model = EGNN(
    hidden_nf=128, 
    out_nf=4, 
    n_layers=2, 
    activation=jax.nn.swish,  
    normalize=True
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    coords = jax.random.normal(key, (BATCH_SIZE, N_NODES, 3)) # Coordinates in R^3 
    node_features = jnp.ones((BATCH_SIZE, N_NODES, 3))    


    edge_attr = jnp.ones((BATCH_SIZE,N_NODES*N_NODES,1)) # 5 edges
    inputs = Batch(x=node_features, h=node_features, edge_attr=edge_attr)
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)