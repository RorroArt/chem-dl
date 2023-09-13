import haiku as hk
from einops import rearrange

from models.model import Model

class MLP_Module(hk.Module):
    def __init__(self, hidden_sizes, output_size, activation):
        super().__init__()
        layers = [hk.Sequential([hk.Linear(size), activation]) for size in hidden_sizes] + [hk.Linear(output_size   )]
        self.model = hk.Sequential(layers)
        
    def __call__(self, key, inputs): return self.model(inputs)

def MLP(hidden_sizes, output_size, activation):
    def model(batch, key):
        module = MLP_Module(hidden_sizes, output_size, activation)
        return module(key, rearrange(batch.x, 'b n c -> b (n c)'))
    
    return Model(model)