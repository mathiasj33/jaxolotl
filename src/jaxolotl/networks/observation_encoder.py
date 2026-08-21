"""Encoders for structured environment observations."""

from abc import abstractmethod
from collections.abc import Callable, Sequence
from math import prod
from typing import NamedTuple, override

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.nn.initializers import Initializer

from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.networks.callable_module import CallableModule
from jaxolotl.networks.conv_net import ConvNet
from jaxolotl.networks.mlp import MLP


def flatten_observation(
    observation: NamedTuple, fields: Sequence[str] | None = None
) -> jax.Array:
    """Flatten and concatenate every array leaf of one observation."""
    leaves = []
    for field in observation._fields:
        if fields is not None and field not in fields:
            continue
        leaf = getattr(observation, field)
        if isinstance(leaf, jax.Array):
            leaves.append(leaf.reshape(-1))
        else:
            raise TypeError(
                f"Observation field {field} is not a JAX array, but {type(leaf)}"
            )
    if not leaves:
        raise ValueError("Cannot encode an observation without array leaves.")
    return jnp.concatenate(leaves).astype(jnp.float32)


class ObservationEncoder[TInput: NamedTuple](CallableModule):
    """Encode one unbatched observation PyTree into a fixed-size feature vector."""

    output_size: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, observation: TInput) -> jax.Array:
        pass


class FlattenMLPEncoder[TInput: NamedTuple](ObservationEncoder[TInput]):
    """Flatten a structured observation and process it with an MLP."""

    mlp: MLP
    fields: list[str] | None
    output_size: int

    def __init__(  # noqa: PLR0913
        self,
        input_spec: ObservationSpec,
        out_size: int,
        hidden_sizes: list[int],
        fields: Sequence[str] | None = None,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = jax.nn.initializers.orthogonal(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        final_layer_activation: bool = True,
        key: jax.Array,
    ):
        in_size = sum(
            prod(spec.shape)
            for field, spec in input_spec.items()
            if fields is None or field in fields
        )
        self.mlp = MLP(
            in_size=in_size,
            out_size=out_size,
            hidden_sizes=hidden_sizes,
            activation=activation,
            weight_init=weight_init,
            bias_init=bias_init,
            final_layer_activation=final_layer_activation,
            key=key,
        )
        self.fields = list(fields) if fields is not None else None
        self.output_size = out_size

    @override
    def __call__(self, observation: TInput) -> jax.Array:
        return self.mlp(flatten_observation(observation, fields=self.fields))


class FieldConvEncoder[TInput: NamedTuple](ObservationEncoder[TInput]):
    """Apply a 2D convolutional network to one named observation field."""

    conv_net: ConvNet
    field: str
    output_size: int

    def __init__(
        self,
        input_spec: ObservationSpec,
        field: str,
        channels: list[int],
        kernel_size: tuple[int, int],
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        *,
        final_layer_activation: bool = True,
        key: jax.Array,
    ):
        if field not in input_spec:
            raise ValueError(f"Field {field} not found in input_spec.")
        field_spec = input_spec[field]
        if len(field_spec.shape) != 3:
            raise ValueError(
                f"Field {field} must have 3D shape (H, W, C), but got {field_spec.shape}."
            )
        self.conv_net = ConvNet(
            obs_shape=field_spec.shape,
            channels=channels,
            kernel_size=kernel_size,
            activation=activation,
            final_layer_activation=final_layer_activation,
            key=key,
        )
        self.field = field
        self.output_size = self.conv_net.output_size

    @override
    def __call__(self, observation: TInput) -> jax.Array:
        return self.conv_net(getattr(observation, self.field))


class ParallelEncoder[TInput: NamedTuple](ObservationEncoder[TInput]):
    """Run multiple encoders over one observation and fuse their representations."""

    branches: tuple[ObservationEncoder[TInput], ...]
    fusion: MLP
    output_size: int

    def __init__(
        self,
        input_spec: ObservationSpec,
        branches: Sequence[Callable[..., ObservationEncoder[TInput]]],
        fusion_hidden_sizes: list[int],
        fusion_out_size: int,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        *,
        key: jax.Array,
    ):
        if not branches:
            raise ValueError("ParallelEncoder requires at least one branch.")

        keys = jax.random.split(key, len(branches) + 1)
        branch_keys, fusion_key = keys[:-1], keys[-1]
        self.branches = tuple(
            branch(input_spec=input_spec, key=branch_key)
            for branch, branch_key in zip(branches, branch_keys, strict=True)
        )
        self.fusion = MLP(
            in_size=sum(branch.output_size for branch in self.branches),
            out_size=fusion_out_size,
            hidden_sizes=fusion_hidden_sizes,
            activation=activation,
            key=fusion_key,
        )
        self.output_size = fusion_out_size

    @override
    def __call__(self, observation: TInput) -> jax.Array:
        encoded = [branch(observation) for branch in self.branches]
        return self.fusion(jnp.concatenate(encoded))
