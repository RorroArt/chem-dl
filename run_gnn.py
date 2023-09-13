import jax 
import jax.numpy as jnp

from models.gnn import GNN 
from models.data import Batch

N_NODES = 5
BATCH_SIZE =10

model = GNN(
    hidden_nf=128, 
    out_nf=4, 
    n_layers=2, 
    activation=jax.nn.swish,  
    normalize=True
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    node_features = jnp.ones((BATCH_SIZE, N_NODES, 3)) # 5 nodes, 3 channels 
    
    edge_attr = jnp.ones((BATCH_SIZE,N_NODES*N_NODES,1)) # 5 edges
    inputs = Batch(x=node_features, edge_attr=edge_attr)
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)