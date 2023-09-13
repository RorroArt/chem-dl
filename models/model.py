import jax 
import jax.numpy as jnp

import haiku as hk

from typing import NamedTuple, Callable, Tuple

# This file contains the abstract classes for all models

# Abstract class for models
class Model:
    __slots__ = '_model'

    def __init__(self, model: hk.Module) -> None:
        self._model = model

    @property
    def model(self):
        return hk.without_apply_rng(hk.transform(self._model))

    def init(self, key, inputs): return self.model.init(key, inputs, key)
    def apply(self, params, key, inputs): return self.model.apply(params, inputs, key)

class Encoder(hk.Module):
    pass

class Decoder(hk.Module):
    pass
    

# Base class for autoencoders and variational autoencoders
class AutoEncoder(Model):
    __slots__ = '_encoder', '_decoder', 'latent_size', 'variational'
    
    def __init__(self, encoder: Encoder, decoder: Decoder, latent_size: int, variational: bool) -> None:
        self._encoder = encoder
        self._decoder = decoder
        self.latent_size = latent_size
        self.variational = variational

    @property
    def encoder(self):
        return hk.transform(self._encoder)

    @property
    def decoder(self):
        return hk.transform(self._decoder)
    
    def encode(self, encoder_params:Tuple[hk.Params], inputs, key):
        return self.encoder.apply(encoder_params, key, inputs, key)
    
    def decode(self, decoder_params:Tuple[hk.Params], latent, key):
        return self.decoder.apply(decoder_params, key, latent, key)
   

    
