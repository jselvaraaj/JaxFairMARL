# test_lexicographic_bottleneck.py

import jax
import jax.numpy as jnp

from envs.bottleneck_assignment_optimization import (
    lexicographic_bottleneck_assignment,
    solve_bottleneck_assignment,
)


def test_lexicographic_bottleneck_assignment():
    with jax.disable_jit():
        C = jnp.array(
            [
                [4, 1, 9],
                [8, 6, 2],
                [5, 7, 3],
            ]
        )
        rows, cols = solve_bottleneck_assignment(C)
        # assert jnp.array_equal(rows, jnp.array([0, 1, 2]))
        assert jnp.array_equal(cols[rows], jnp.array([1, 2, 0]))

        rows, cols = lexicographic_bottleneck_assignment(C)
        # assert jnp.array_equal(rows, jnp.array([2, 0, 1]))
        assert jnp.array_equal(cols[rows], jnp.array([1, 2, 0]))


def test_lexicographic_bottleneck_assignment_2():
    with jax.disable_jit():
        C = jnp.array(
            [
                [4, 4, 1],
                [4, 9, 4],
                [6, 6, 7],
            ]
        )
        rows, cols = solve_bottleneck_assignment(C)
        # assert jnp.array_equal(rows, jnp.array([0, 1, 2]))
        assert jnp.array_equal(cols[rows], jnp.array([0, 2, 1]))

        rows, cols = lexicographic_bottleneck_assignment(C)
        # assert jnp.array_equal(rows, jnp.array([2, 0, 1]))
        assert jnp.array_equal(cols[rows], jnp.array([2, 0, 1]))
