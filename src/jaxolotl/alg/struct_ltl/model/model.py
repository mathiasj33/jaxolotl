from typing import override

import hydra
import jax
import jax.numpy as jnp
from equinox import nn
from jaxtyping import PyTree
from omegaconf import DictConfig

from jaxolotl.alg.struct_ltl.reach_avoid.jax_clause_reach_avoid_sequence import (
    JaxClauseReachAvoidSequence,
)
from jaxolotl.environments.observation_spec import ObservationSpec
from jaxolotl.environments.spaces import Space
from jaxolotl.networks.deep_sets import DeepSets
from jaxolotl.networks.observation_encoder import ObservationEncoder
from jaxolotl.networks.sequence_encoder import SequenceEncoder
from jaxolotl.rl.actor_critic import ActorCritic


class StructLTLModel(ActorCritic):
    env_net: ObservationEncoder
    embedding: nn.Embedding
    neg_linear: nn.Linear
    clause_mlp: DeepSets
    disjunct_mlp: DeepSets
    sequence_encoder: SequenceEncoder

    def __init__(
        self,
        obs_spec: ObservationSpec,
        act_space: Space,
        num_propositions: int,
        key: jax.Array,
        **kwargs,
    ):
        config = DictConfig(kwargs)
        key, env_key = jax.random.split(key)
        self.env_net = hydra.utils.instantiate(
            config.env_net, input_spec=obs_spec, key=env_key, _convert_="object"
        )
        key, emb_key = jax.random.split(key)
        embedding_dim = config.sequence.embedding_dim
        self.embedding = nn.Embedding(
            num_embeddings=num_propositions + 1,  # +1 for epsilon transitions
            embedding_size=embedding_dim,
            key=emb_key,
        )
        key, neg_key = jax.random.split(key)
        self.neg_linear = nn.Linear(
            in_features=embedding_dim,
            out_features=embedding_dim,
            key=neg_key,
        )
        key, clause_key = jax.random.split(key)
        self.clause_mlp = hydra.utils.instantiate(
            config.sequence.clause_mlp,
            embedding_dim=embedding_dim,
            key=clause_key,
        )
        key, disjunct_key = jax.random.split(key)
        self.disjunct_mlp = hydra.utils.instantiate(
            config.sequence.disjunct_mlp,
            embedding_dim=embedding_dim,
            key=disjunct_key,
        )

        sequence_dim = 2 * config.sequence.disjunct_mlp.out_size

        key, encoder_key = jax.random.split(key)
        self.sequence_encoder = SequenceEncoder(
            input_size=sequence_dim,
            hidden_size=2 * embedding_dim,
            mode=config.sequence.encoder,
            attention_config=getattr(config.sequence, "attention", None),
            key=encoder_key,
        )

        joint_dim = self.env_net.output_size + self.sequence_encoder.output_size
        super().__init__(config, act_space, joint_dim, key)

    @override
    def _get_actor_context(self, obs: PyTree) -> PyTree:
        return obs.epsilon_mask

    @override
    def _compute_common_features(self, obs: PyTree) -> jax.Array:
        x = jax.vmap(self.env_net)(obs.features)
        emb = jax.vmap(self._compute_sequence_embedding)(obs.seq)
        return jnp.concatenate([x, emb], axis=-1)

    def _compute_sequence_embedding(
        self, seq: JaxClauseReachAvoidSequence
    ) -> jax.Array:
        def embed_clause(indices: jax.Array, neg_mask: jax.Array) -> jax.Array:
            # indices shape: (num_propositions,)
            mask = indices != -1
            embeddings = jax.vmap(self.embedding)(indices * mask)
            # embeddings shape: (num_propositions, embedding_dim)
            neg_embeddings = jax.vmap(self.neg_linear)(embeddings)
            embeddings = jnp.where(neg_mask[:, None], neg_embeddings, embeddings)
            embeddings = embeddings * mask[:, None]  # zero out padding embeddings
            return self.clause_mlp(embeddings)  # shape: (out_size,)

        def embed_disjunct(
            clauses: jax.Array, neg_masks: jax.Array, num_clauses: jax.Array
        ) -> jax.Array:
            # clauses shape: (max_clauses, num_propositions)
            clause_embeddings = jax.vmap(embed_clause)(clauses, neg_masks)
            mask = jnp.arange(clauses.shape[0]) < num_clauses
            clause_embeddings = clause_embeddings * mask[:, None]
            # clause_embeddings shape: (max_clauses, clause_mlp.out_size)
            return self.disjunct_mlp(clause_embeddings)  # shape: (out_size,)

        reach_emb = jax.vmap(embed_clause)(seq.reach_clauses, seq.reach_negatives)
        avoid_emb = jax.vmap(embed_disjunct)(
            seq.avoid_clauses, seq.avoid_negatives, seq.num_avoid_clauses
        )
        reach_avoid = jnp.concatenate([reach_emb, avoid_emb], axis=-1)
        # reach_avoid shape: (max_seq_length, 2 * disjunct_mlp.out_size)

        return self.sequence_encoder(reach_avoid, seq.depth)
