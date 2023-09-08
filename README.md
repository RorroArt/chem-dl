# Chemistry Deep Learning

This is a collection of deep learning models with applications in chemistry. All models are implemented using DeepMind's haiku framework which is built on top of jax (so you can run them on TPUs). 

## Installation

- First clone this repository.

- Install jax and jaxlib following the detailed instructions in the  [official website](https://jax.readthedocs.io/en/latest/#installation) and make sure to add the correct gpu support for your system.

- Install haiku, optax and einsum by running `pip install git+https://github.com/deepmind/dm-haiku optax einsum`

- Finally download the data by running `python get_qm9.py`

- If you want to use the Equivariant MLPs, you need to install the emlp library running `pip install emlp`

## Models

### MLP
The classical multilayer perceptron can be very useful for some chemistry applications, and that is why I included it on this repo. The usage is very simple, to try it out run the following code:

```
import jax 
import jax.numpy as jnp

from models import Model
from models.mlp import MLP 
from models.data import Batch

model = Model(
    MLP(
        hidden_sizes = [128, 128],
        output_size = 10,
        activation = jax.nn.relu
    )
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    inputs = Batch(x=jnp.ones((1, 784)))
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)
    >>> (1, 10)
```

### EMLP

The problem with normal multilayer perceptrons is that they don't take advantage of the geometrical symmetries of the data, and chemistry datasets are filled with this. A good way of exploiting this translational, rotational, and permutation symmetries if by enforcing equivariance in the models `[1]`. Turns out that you can easily construct Equivariant MLPs by following the method described in this [paper]() `[2]`. In this repo we implemented these models using the amazing library the authors built. Try out an SO(3) equivariant model by running the following code:

```
import jax 
import jax.numpy as jnp

from models import Model
from models.emlp import EMLP 
from models.data import Batch

from emlp.reps import T
from emlp.groups import SO

model = Model(
    model = EMLP(
        rep_in = 2*T(1),
        rep_out = T(1),
        group = SO(3),
        ch = 128,
        num_layers = 2
    )
)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    inputs = Batch(x=jnp.ones((1, 6)))
    params = model.init(key, inputs)
    y = model.apply(params, key, inputs)
    print(y.shape)
    >>> (1, 3)
```

# GNN
Another cool way of exploiting the symmetries is by taking advantage of the fact that most of this models can be represented as graphs! So, we can use graph neural networks to achieve this. For chemical data, usually only node features such as the coordinates, masses, and velocities of the atoms are considered. Hence, you only need this information and a description of how the nodes are connected. You can test it out running the following code:

```

```

# EGNN
But is there a way of combining this two ideas (equivariance and graph representations)? It turns out that yes! We can construct a E(n) Equivariant GNN using the ideas from this [paper](). The representations are very simple (relative distances) and this ensures equivariance in n-dimensioanl euclidean data. You can test it out running the following code:

```

```


## Benchmarks

There are currently two tasks in which you can benchmark this models: 

- QM9 HOMO prediction (Supervised)
- QM9 Relative distances prediction (Self-supervised)

To run a benchmark simply execute: 

```
python run_<model-name>.py
```

Replacing `<model-name>` as desired. Results will be saved in a json file that you can find in the `./results` directory

For purposes of comparinson, all models have approximately Xk parameters.

### HOMO Prediction results

Paste the full task results here

Paste the sample efficiency results here

### Relative Distances results

Paste the full task results here

Paste the sample efficiency results here

## References 



