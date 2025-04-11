from functools import partial

import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(0,))
def sample_points(num_points, key, min_dist_between_points, bounds=(0, 1)):
    def body_fun(state):
        key, points, num_accepted = state
        key, subkey = jax.random.split(key)
        new_point = jax.random.uniform(subkey, (2,), minval=bounds[0], maxval=bounds[1])
        distances = jnp.sqrt(jnp.sum((points - new_point) ** 2, axis=1))

        # Create a boolean mask indicating which rows (points) are accepted
        # i.e., from index 0 up to num_accepted-1.
        mask = jnp.arange(num_points) < num_accepted

        # "Ignore" distances for unaccepted slots by setting them to +inf
        # so they won't affect the minimum-dist checks.
        distances = jnp.where(mask, distances, jnp.inf)

        is_valid = jnp.all(distances >= min_dist_between_points) | (num_accepted == 0)

        points = jax.lax.dynamic_update_slice(
            points, jnp.expand_dims(new_point, 0), (num_accepted, 0)
        )
        num_accepted += is_valid

        return key, points, num_accepted

    init_points = jnp.zeros((num_points, 2))
    init_state = (key, init_points, 0)

    final_state = jax.lax.while_loop(
        lambda state: state[2] < num_points, body_fun, init_state
    )

    return final_state[1]
