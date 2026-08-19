from collections.abc import Callable
from typing import override

import equinox as eqx
import jax
import jax.numpy as jnp
import jraph
from equinox import nn
from jax.nn.initializers import Initializer

from jaxolotl.alg.ltl2action.utils.jax_formula_closure import JaxFormulaGraph
from jaxolotl.networks.callable_module import CallableModule
from jaxolotl.networks.network_utils import make_linear


class RGCN(CallableModule):
    """A Relational Graph Convolutional Network (RGCN)."""

    # we share layers across message passing steps for each relation type, following
    # the LTL2Action paper.
    linears: list[nn.Linear]
    self_loop: nn.Linear
    num_layers: int
    activation: Callable[[jax.Array], jax.Array]

    def __init__(
        self,
        in_size: int,
        out_size: int,
        num_layers: int,
        num_relations: int,
        activation: Callable[[jax.Array], jax.Array] = jax.nn.relu,
        weight_init: Initializer | None = jax.nn.initializers.glorot_uniform(),
        bias_init: Initializer | None = jax.nn.initializers.zeros,
        *,
        key: jax.Array,
        **kwargs,
    ):
        del kwargs
        if in_size != out_size:
            raise ValueError("RGCN requires in_size == out_size for weight sharing.")
        key, self_loop_key = jax.random.split(key)
        self.self_loop = make_linear(
            in_size + out_size,
            out_size,
            weight_init,
            bias_init,
            key=self_loop_key,
        )
        linear_keys = jax.random.split(key, num_relations)
        self.linears = [
            make_linear(
                in_size + out_size,  # concatenate input and original embeddings
                out_size,
                weight_init,
                bias_init,
                key=linear_keys[i],
            )
            for i in range(num_relations)
        ]
        self.activation = activation
        self.num_layers = num_layers

    def _update(
        self, graph: JaxFormulaGraph, features: jax.Array, original_features: jax.Array
    ) -> jax.Array:
        """Performs a single RGCN message-passing step."""
        # 1. Setup Masks and Shapes
        node_mask = graph.node_mask
        edge_mask = graph.edge_mask
        total_num_nodes = graph.nodes.shape[0]

        # We mask input features to ensure clean padding
        features = jnp.concatenate([features, original_features], axis=-1)
        features = jnp.where(node_mask[:, None], features, 0.0)
        new_features = jnp.zeros_like(original_features)

        # 2. Updates for each relation type
        for rel_type, layer in enumerate(self.linears):
            rel_edge_mask = jnp.logical_and(edge_mask, (graph.edge_types == rel_type))
            node_degree = jraph.segment_sum(
                rel_edge_mask.astype(jnp.float32),
                graph.edges[:, 1],
                num_segments=total_num_nodes,
            )
            messages = features[graph.edges[:, 0]]
            messages = jnp.where(rel_edge_mask[:, None], messages, 0.0)
            aggr = jraph.segment_sum(
                messages, graph.edges[:, 1], num_segments=total_num_nodes
            )
            norm = aggr / node_degree[:, None].clip(min=1.0)
            updated_features = jax.vmap(layer)(norm)
            new_features += updated_features

        # 3. Self-Loop Contribution
        new_features += jax.vmap(self.self_loop)(features)
        return new_features

    @override
    @eqx.filter_jit
    def __call__(self, graph: JaxFormulaGraph, features: jax.Array) -> jax.Array:
        """Processes the graph through all RGCN layers.

        Args:
            graph: A single (non-batched) JaxFormulaGraph.
            features: Initial node features (shape: [num_nodes, in_size]).

        Returns:
            features: Final node features (shape: [num_nodes, out_size]).
        """

        def layer(_, current_features):
            current_features = self._update(graph, current_features, features)
            current_features = self.activation(current_features)
            current_features = jnp.where(
                graph.node_mask[:, None], current_features, 0.0
            )
            return current_features

        return jax.lax.fori_loop(0, self.num_layers, layer, features, unroll=True)
