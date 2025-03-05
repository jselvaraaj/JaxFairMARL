import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array, Float

from envs.target_mpe_env import MPEState, TargetMPEEnvironment

pos_dim = "pos_dim"
num_entities = "num_entities"
num_agents = "num_agents"
num_landmarks = "num_landmarks"


def get_direction_vector(
    vec1: Float[Array, f"{num_agents} {pos_dim}"],
    vec2: Float[Array, f"{num_agents} {pos_dim}"],
) -> Float[Array, f"{num_agents} {pos_dim}"]:
    dist_traveled = vec2 - vec1
    dist_traveled = dist_traveled / jnp.linalg.norm(dist_traveled, axis=1)[:, None]
    return dist_traveled


@pytest.fixture
def env_setup():
    max_steps = 10
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key, 2)
    num_agents = 3
    env = TargetMPEEnvironment(num_agents=num_agents)
    initial_communication_message = jnp.asarray([])
    initial_entity_position = jnp.asarray([])

    observation, graph, state = env.reset(
        key_r, initial_communication_message, initial_entity_position
    )

    return env, graph, state, max_steps, key


@pytest.fixture
def init_state(env_setup):
    env, _, _, _, key = env_setup
    key_agent, key_landmark = jax.random.split(key)

    assert env.num_agents == 3, "This test only works for 3 agents"

    entity_positions = jnp.concatenate(
        [
            jnp.asarray([[0.0, 0.0], [0.0, 200.0], [200.0, 0.0]]),
            jnp.asarray([[1000.0, 1000.0], [1000.0, 1200.0], [1200.0, 1000.0]]),
        ]
    )

    return MPEState(
        dones=jnp.full(env.num_agents, False),
        step=0,
        entity_positions=entity_positions,
        entity_velocities=jnp.zeros((env.num_entities, env.position_dim)),
        agent_indices_to_landmark_index=jnp.arange(env.num_agents, dtype=jnp.int32),
        landmark_occupancy=jnp.zeros(env.num_landmarks, dtype=jnp.int32),
        closest_landmark_idx=jnp.zeros(env.num_agents, dtype=jnp.int32),
        distance_travelled=jnp.zeros(env.num_agents, dtype=jnp.float32),
        did_agent_die_this_time_step=jnp.zeros(env.num_agents, dtype=jnp.bool_),
        agent_communication_message=jnp.asarray([]),
        agent_visibility_radius=jnp.ones(env.num_agents, dtype=jnp.float32)
        * env.agent_visibility_radius,
    )


def test_do_nothing_action(env_setup, init_state):
    """Test that the target mpe do nothing discrete action works."""
    with jax.disable_jit(True):
        env, graph, state, max_steps, key = env_setup
        state = init_state
        graph = env.get_graph(state)
        prev_state = state
        initial_communication_message = jnp.asarray([])
        initial_entity_position = jnp.asarray([])

        for _ in range(max_steps):
            key, key_env = jax.random.split(key)
            action = {agent_label: 0 for i, agent_label in enumerate(env.agent_labels)}

            obs, _, state, rew, dones, _ = env.step(
                key_env,
                state,
                action,
                initial_communication_message,
                initial_entity_position,
            )

            assert jnp.array_equal(
                prev_state.entity_positions, state.entity_positions
            ), f"{prev_state.entity_positions} != {state.entity_positions}"
            prev_state = state


@pytest.mark.parametrize(
    "action_num,expected_direction",
    [
        (1, jnp.asarray([-1.0, 0.0])),  # left
        (2, jnp.asarray([1.0, 0.0])),  # right
        (3, jnp.asarray([0.0, -1.0])),  # down
        (4, jnp.asarray([0.0, 1.0])),  # up
    ],
)
def test_directional_actions(env_setup, init_state, action_num, expected_direction):
    """Test that the target mpe directional actions work correctly."""
    env, _, state, max_steps, key = env_setup
    state = init_state
    initial_communication_message = jnp.asarray([])
    initial_entity_position = jnp.asarray([])

    init_state_agent_positions = state.entity_positions[: env.num_agents]

    for _ in range(max_steps):
        key, key_env = jax.random.split(key)
        action = {agent_label: action_num for agent_label in env.agent_labels}

        obs, _, state, rew, dones, _ = env.step(
            key_env,
            state,
            action,
            initial_communication_message,
            initial_entity_position,
        )

    state_agent_positions = state.entity_positions[: env.num_agents]
    dist_travelled = get_direction_vector(
        init_state_agent_positions, state_agent_positions
    )

    assert jnp.allclose(dist_travelled, expected_direction[None], atol=1e-6)


# if __name__ == "__main__":
#     pytest.main([__file__])
