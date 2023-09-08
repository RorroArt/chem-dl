import jax 
import jax.numpy as jnp

from models import Model 
from models.mlp import MLP 
from models.data import Batch

model = MLP(
    hidden_sizes = [128, 128],
    output_size = 10,
    activation = jax.nn.relu
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    inputs = Batch(x=jnp.ones((2, 784)))
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)