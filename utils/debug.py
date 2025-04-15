import jax
import jax.numpy as jnp


def print_if_nonfinite(x, name=None):
    is_finite = jnp.all(jnp.stack([jnp.isfinite(x).all() for x in jax.tree.leaves(x)]))

    def true_fn(x):
        pass

    def false_fn(x):
        jax.tree.map_with_path(
            lambda path, x: jax.debug.print(
                "{}:: {}: {}", name, jax.tree_util.keystr(path), jnp.isfinite(x).all()
            ),
            x,
        )

    jax.lax.cond(is_finite, true_fn, false_fn, x)
