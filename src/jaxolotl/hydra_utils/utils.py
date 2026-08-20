import jax
from omegaconf import OmegaConf


def register_custom_resolvers():
    activation_functions = {
        "relu": jax.nn.relu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "softplus": jax.nn.softplus,
    }
    OmegaConf.register_new_resolver("act", lambda name: activation_functions[name])
