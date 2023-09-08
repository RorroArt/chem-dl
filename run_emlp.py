import jax 
import jax.numpy as jnp

from models import Model
from models.emlp import EMLP 
from models.data import Batch

from emlp.reps import T
from emlp.groups import SO

model = EMLP(
    rep_in = 2*T(1),
    rep_out = T(1),
    group = SO(3),
    ch = 128,
    num_layers = 2
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    inputs = Batch(x=jnp.ones((1, 6)))
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)