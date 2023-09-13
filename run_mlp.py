import jax 
import jax.numpy as jnp
from jax import random

from models.mlp import MLP 
from models.data import Batch

from torch_geometric.datasets import QM9
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

import copy
import time

import optax

BATCH_SIZE = 50
SAMPLES = 10000
TEST_SAMPLES = 10000
EPOCHS = int(100000 / SAMPLES) # Always training for 100000 steps
LEARNING_RATE = 1e-3


class MyTransform:
    def __call__(self, data):
        data = copy.copy(data)
        data.y = data.y[:, 2]  # Specify target. (HOMO)
        return data

path = './data/qm9'
transform = T.Compose([MyTransform(), T.Pad(max_num_nodes=29)])
dataset = QM9(path, transform=transform).shuffle()


train_dataset = dataset[:SAMPLES]
test_dataset = dataset[SAMPLES:SAMPLES+TEST_SAMPLES]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = MLP(
    hidden_sizes = [128, 128],
    output_size = 1,
    activation = jax.nn.relu
)

def torch2jax(torch_batch):
    x = jnp.array(torch_batch.pos.view(BATCH_SIZE, 29, 3).numpy()) # Coordinates
    h = jnp.array(torch_batch.x.view(BATCH_SIZE, 29, 11).numpy()) # Node features 
    x = jnp.concatenate([x,h], axis=2)

    y = jnp.array(torch_batch.y.numpy())
    return Batch(x=x), y

def loss_fn(params, key, x, y):
    y_hat = model.apply(params, key, x)
    return jnp.abs(y_hat.reshape(y.shape) - y).mean()


def build_update_function(optimizer, loss_fn):
  @jax.jit
  def update(params, key, state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, key, x, y)
    updates, state = optimizer.update(grads, state, params)
    params = optax.apply_updates(params, updates)
    return params, state, loss
  return update



optim = optax.adam(LEARNING_RATE)

if __name__ == '__main__':
    key = jax.random.PRNGKey(0)

    fake_inputs,_ = torch2jax(next(iter(train_loader)))
    params = model.init(key, fake_inputs)
    
    state = optim.init(params)
    update = build_update_function(optim, loss_fn)
    
    all_losses = []

    for i in range(EPOCHS):
        losses = []
        start = time.time()
        loss = 0
        for j, batch in enumerate(train_loader):
            key, subkey = random.split(key)
            x, y = torch2jax(batch)
            params, state, loss = update(params, key, state, x, y)
            losses.append(loss.item())
        end = time.time()
        total_loss = jnp.array(losses).mean()
        all_losses.append(total_loss.item())
        print(f'Epoch: {i} - loss: {total_loss: .3f} - Execution time: {(end-start): .3f} sec \n')

    test_losses = []
    for j, batch in enumerate(test_loader):
        key, subkey = random.split(key)
        x, y = torch2jax(batch)
        loss = loss_fn(params, key, x, y)
        test_losses.append(loss.item())

    total_loss = jnp.array(test_losses).mean()
    print(f'Test loss {total_loss: .3f}')