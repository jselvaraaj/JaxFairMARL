import functools
from typing import Literal

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from einshape import jax_einshape as einshape
from flax.linen.initializers import constant, orthogonal
from jaxtyping import Array, Float
from jraph._src import utils

from config.mappo_config import MAPPOConfig


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# noinspection DuplicatedCode
class ActorRNN(nn.Module):
    action_dim: list[int]
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config.network_config.fc_dim_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        if self.config.network_config.use_rnn:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        for _ in range(self.config.network_config.actor_num_hidden_linear_layer - 1):
            embedding = nn.Dense(
                self.config.network_config.gru_hidden_dim,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.config.network_config.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = embedding

        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        action_logits = jnp.clip(action_logits, -10, 10)
        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


# This is built off of the GAT implementation in jraph
class GraphMultiHeadAttentionLayer(nn.Module):
    config: MAPPOConfig

    @functools.partial(nn.jit, static_argnames=("avg_multi_head",))
    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, avg_multi_head):
        # Assumes that given graph is in into jraph compatible format
        nodes, edges, receivers, senders, _, _, _ = graph

        # Equivalent to the sum of n_node, but statically known.
        sum_n_node = nodes.shape[0]

        def linear_layer(x):
            for _ in range(self.config.network_config.graph_num_linear_layer - 1):
                x = nn.Dense(
                    self.config.network_config.graph_hidden_feature_dim,
                    kernel_init=orthogonal(jnp.sqrt(2)),
                    bias_init=constant(0.0),
                )(x)
                x = nn.relu(x)
            x = nn.Dense(
                self.config.network_config.graph_hidden_feature_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)
            x = nn.relu(x)
            return x

        def key_projection(x):
            return nn.Dense(
                self.config.network_config.graph_attention_key_dim,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)

        # embed node features
        nodes = linear_layer(nodes)
        # embed edge features
        edge_features = linear_layer(edges)

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        nodes_seg_sum_from_each_attn_head = []

        for _ in range(self.config.network_config.num_heads_per_attn_layer):
            # extract key for node feature to be used in attention
            key_sent_attributes = key_projection(sent_attributes)
            key_received_attributes = key_projection(received_attributes)

            key_edge_features = key_projection(edge_features)

            key_received_attributes = key_received_attributes + key_edge_features

            softmax_logits: Float[Array, Literal["edge_id"]] = jnp.sum(
                key_sent_attributes * key_received_attributes, axis=-1
            ) / jnp.sqrt(self.config.network_config.graph_attention_key_dim)

            # Compute the softmax weights on the entire tree.
            """
            Let's say we have the following graph:
            - 6 nodes with indices [0, 1, 2, 3, 4, 5]
            - following are the edges:
                - 0 -> 1
                - 0 -> 2
                - 1 -> 3
                - 1 -> 4
                - 2 -> 5
            - total number of possible edges = 6C2 = 15
            - senders = [0, 0, 1, 1, 2] + [-1] * (15 - 5) # python list addition
            - receivers = [1, 2, 3, 4, 5] + [-1] * (15 - 5) # python list addition
            helpful resource: https://github.com/google-deepmind/educational/blob/master/colabs/summer_schools/intro_to_graph_nets_tutorial_with_jraph.ipynb
            """
            # softmax_logits.shape = (num_edges,) = (15,)
            # in this case sum_n_node = num_nodes = 6
            weights = utils.segment_softmax(
                softmax_logits, segment_ids=receivers, num_segments=sum_n_node
            )
            # weights.shape = (num_edges,) = (15,)
            # Apply weights
            messages = weights[..., None] * sent_attributes
            # Aggregate messages to nodes.
            nodes_seg_sum = jax.ops.segment_sum(
                messages, receivers, num_segments=sum_n_node
            )
            nodes_seg_sum_from_each_attn_head.append(nodes_seg_sum)

        if avg_multi_head:
            nodes = jnp.mean(jnp.stack(nodes_seg_sum_from_each_attn_head), axis=0)
        else:
            nodes = jnp.concatenate(nodes_seg_sum_from_each_attn_head, axis=-1)
        return graph._replace(nodes=nodes)


class GraphStackedMultiHeadAttention(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        """Applies a Graph Attention layer."""

        # Make the given graph into jraph compatible format
        (
            equivariant_nodes,
            non_equivariant_nodes,
            edges,
            receivers,
            senders,
            _,
            n_node,
            n_edge,
            _,
        ) = graph

        (
            num_env_steps,
            num_actors,
            num_nodes_in_one_graph,
            num_time_steps_concatenated,
            *equivariant_node_feature_dim,
        ) = equivariant_nodes.shape
        (
            _,
            _,
            _,
            _,
            *non_equivariant_node_feature_dim,
        ) = non_equivariant_nodes.shape
        _, _, num_edges, edge_feature_dim = edges.shape
        # flattening the batch dimension. so one graph contains all the time steps for all the actors and edges will be the only source of information for which node is connected to which other node.
        # That is, as long as there are no edges across these batch dimensions, this should be a safe operation.
        num_graph = num_env_steps * num_actors
        equivariant_nodes = equivariant_nodes.reshape(
            (
                num_graph * num_nodes_in_one_graph,
                num_time_steps_concatenated,
                *equivariant_node_feature_dim,
            )
        )
        non_equivariant_nodes = non_equivariant_nodes.reshape(
            (
                num_graph * num_nodes_in_one_graph,
                num_time_steps_concatenated,
                *non_equivariant_node_feature_dim,
            )
        )
        edges = edges.reshape((num_graph * num_edges, edge_feature_dim))

        # because we are combining multiple graphs into one, we need to make sure that the node indices are unique for each of the original graph.
        index_offset = jnp.arange(num_graph).reshape(num_env_steps, num_actors)[
            ..., None
        ]
        receivers += index_offset * num_nodes_in_one_graph
        senders += index_offset * num_nodes_in_one_graph
        receivers = receivers.flatten()
        senders = senders.flatten()
        n_node = n_node.flatten()
        n_edge = n_edge.flatten()

        # for now do non equivariant transformation and also concate the stacked node features
        equivariant_nodes = einshape("gtnf->g(tnf)", equivariant_nodes)
        non_equivariant_nodes = einshape("gtf->g(tf)", non_equivariant_nodes)
        # Embed entity_type.
        entity_type = non_equivariant_nodes[..., -1].astype(jnp.int32)
        entity_emb = nn.Embed(
            self.config.derived_values.num_entity_types,
            self.config.network_config.entity_type_embedding_dim,
        )(entity_type)
        non_equivariant_nodes = jnp.concatenate(
            [non_equivariant_nodes[..., :-1], entity_emb], axis=-1
        )
        nodes = jnp.concatenate([equivariant_nodes, non_equivariant_nodes], axis=-1)

        graph = jraph.GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            n_node=n_node,
            n_edge=n_edge,
            globals=None,
        )

        for _ in range(self.config.network_config.num_graph_attn_layers - 1):
            graph = GraphMultiHeadAttentionLayer(self.config)(
                graph, avg_multi_head=False
            )
        # Average the multi-head attention for the last layer
        graph = GraphMultiHeadAttentionLayer(self.config)(graph, avg_multi_head=True)

        equivariant_nodes, edges, receivers, senders, _, n_node, n_edge = graph

        # note the other elements in the graph are still in jraph compatible format
        # but not reverting it back since won't be using it anymore
        equivariant_nodes = equivariant_nodes.reshape(
            num_env_steps, num_actors, num_nodes_in_one_graph, -1
        )
        graph = graph._replace(nodes=equivariant_nodes)
        return graph


class GraphAttentionActorRNN(nn.Module):
    action_dim: list[int]
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):
        _, graph, dones = x

        agent_indices = graph.agent_indices

        if self.config.network_config.use_graph_attention_in_actor:
            graph_embedding = GraphStackedMultiHeadAttention(self.config)(graph)
            equivariant_nodes = graph_embedding.nodes
            nodes = equivariant_nodes
        else:
            equivariant_nodes = graph.equivariant_nodes.reshape(
                graph.equivariant_nodes.shape[:-2] + (-1,)
            )
            # concate the equivariant features
            equivariant_nodes = einshape("...tnf->...t(nf)", graph.equivariant_nodes)

            nodes = jnp.concatenate(
                [equivariant_nodes, graph.non_equivariant_nodes], axis=-1
            )
            # Embed entity_type.
            entity_type = nodes[..., -1].astype(jnp.int32)
            entity_emb = nn.Embed(
                self.config.derived_values.num_entity_types,
                self.config.network_config.entity_type_embedding_dim,
            )(entity_type)
            nodes = jnp.concatenate([nodes[..., :-1], entity_emb], axis=-1)

            # concate the stacked observation
            nodes = einshape("...tf->...(tf)", nodes)

        agent_node_features = nodes[
            jnp.arange(nodes.shape[0])[..., None],
            jnp.arange(nodes.shape[1])[None, ...],
            agent_indices,
        ]
        obs = jnp.concatenate([agent_node_features], axis=-1)

        hidden, pi = ActorRNN(self.action_dim, self.config)(hidden, (obs, dones))

        return hidden, pi


# noinspection DuplicatedCode
class CriticRNN(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):
        _w_s, graph, dones = x

        if self.config.network_config.use_graph_attention_in_critic:
            # this is target mpe specific
            agent_indices = graph.agent_indices

            num_agents = self.config.env_config.env_kwargs.num_agents
            num_entities = 2 * num_agents
            # Full observatibility, agent can see all other agents and landmarks
            senders, receivers = jnp.meshgrid(
                jnp.arange(num_entities), jnp.arange(num_agents)
            )
            senders = senders.flatten()
            receivers = receivers.flatten()
            senders = jnp.broadcast_to(
                senders, graph.senders.shape[:-1] + senders.shape
            )
            receivers = jnp.broadcast_to(
                receivers, graph.receivers.shape[:-1] + receivers.shape
            )

            # Assume edge features are 0
            edges = jnp.zeros(graph.edges.shape[:-2] + (senders.shape[-1], 1))

            graph = graph._replace(senders=senders, receivers=receivers, edges=edges)

            graph_embedding = GraphStackedMultiHeadAttention(self.config)(graph)
            nodes = graph_embedding.nodes

            world_state = nodes[
                jnp.arange(nodes.shape[0])[..., None],
                jnp.arange(nodes.shape[1])[None, ...],
                agent_indices,
            ]
        else:
            # concate the equivariant features
            equivariant_nodes = einshape("...tnf->...(tnf)", graph.equivariant_nodes)
            non_equivariant_nodes = einshape(
                "...tf->...(tf)", graph.non_equivariant_nodes
            )
            nodes = jnp.concatenate([equivariant_nodes, non_equivariant_nodes], axis=-1)
            # Embed entity_type.
            entity_type = nodes[..., -1].astype(jnp.int32)
            entity_emb = nn.Embed(
                self.config.derived_values.num_entity_types,
                self.config.network_config.entity_type_embedding_dim,
            )(entity_type)
            nodes = jnp.concatenate([nodes[..., :-1], entity_emb], axis=-1)

            world_state = jnp.sum(
                nodes, axis=2
            )  # Aggregate all node features for a given actor and time step
            world_state = jnp.concatenate([world_state, _w_s], axis=-1)

        embedding = nn.Dense(
            self.config.network_config.fc_dim_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        if self.config.network_config.use_rnn:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN()(hidden, rnn_in)

        for _ in range(self.config.network_config.critic_num_hidden_linear_layer - 1):
            embedding = nn.Dense(
                self.config.network_config.gru_hidden_dim,
                kernel_init=orthogonal(2),
                bias_init=constant(0.0),
            )(embedding)
            embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.config.network_config.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        embedding = nn.relu(embedding)

        critic = embedding
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)
