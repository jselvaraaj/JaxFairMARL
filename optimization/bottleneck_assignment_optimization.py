import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float, Int

n_axis = "n"
m_axis = "m"


def solve_bottleneck_assignment(
    cost_matrix: Float[Array, f"{m_axis} {n_axis}"]
) -> tuple[Int[Array, f"{m_axis}"], Int[Array, f"{n_axis}"]]:
    max_search_values = jnp.sort(cost_matrix.flatten())

    low = 0
    high = max_search_values.shape[0] - 1
    # Initialize best_assignment with the same shape as what will be returned
    m, n = cost_matrix.shape
    init_rows = jnp.full((m,), jnp.nan, dtype=jnp.float32)
    init_cols = jnp.full((n,), jnp.nan, dtype=jnp.float32)
    best_assignment = (init_rows, init_cols)

    def cond_fn(state):
        low, high, _ = state
        return low <= high

    # Binary search for the maximum cost assignment. Implicitly, minimized by the binary search condition to decrease the search value if a feasible assignment is found.
    # This is essentially a min max for the linear assignment problem.
    def body_fn(state):
        low, high, best = state
        mid = (low + high) // 2
        mid_value = max_search_values[mid]
        feasible_edges = (cost_matrix <= mid_value).astype(jnp.float32)
        rows, cols = optax.assignment.hungarian_algorithm(-feasible_edges)
        is_feasible = jnp.all(feasible_edges[rows, cols])

        new_low = jnp.where(is_feasible, low, mid + 1)
        new_high = jnp.where(is_feasible, mid - 1, high)
        new_best = jax.lax.cond(
            is_feasible,
            lambda: (rows.astype(jnp.float32), cols.astype(jnp.float32)),
            lambda: best,
        )
        return (new_low, new_high, new_best)

    _, _, best_assignment = jax.lax.while_loop(
        cond_fn, body_fn, (low, high, best_assignment)
    )
    return jax.tree.map(lambda x: x.astype(jnp.int32), best_assignment)


def lexicographic_bottleneck_assignment(
    cost_matrix: Float[Array, f"{m_axis} {n_axis}"]
) -> tuple[Int[Array, f"{m_axis}"], Int[Array, f"{n_axis}"]]:
    # Get shape information
    m_size = cost_matrix.shape[0]

    # Initialize arrays to store results
    row_indices = jnp.full((m_size,), jnp.nan, dtype=jnp.float32)
    col_indices = jnp.full((m_size,), jnp.nan, dtype=jnp.float32)

    cost_matrix = cost_matrix.astype(jnp.float32)

    # Define the scan function
    def scan_fn(carry, idx):
        matrix, rows, cols = carry

        # Find bottleneck assignment
        row_idx, col_idx = solve_bottleneck_assignment(matrix)
        assignment_cost = jnp.max(matrix[row_idx, col_idx])

        # Update matrix with threshold
        matrix = jnp.where(matrix <= assignment_cost, matrix, jnp.inf)

        # Find max cost assignment index
        i = jnp.argmax(matrix[row_idx, col_idx])
        r = row_idx[i]
        c = col_idx[i]

        # Update rows and cols arrays
        rows = rows.at[idx].set(r)
        cols = cols.at[idx].set(c)

        # Set the assignment cost to 0 to indicate that the assignment has been made. Choosing any another assignemnt than this will result in inf cost.
        matrix = matrix.at[r].set(jnp.inf)
        matrix = matrix.at[:, c].set(jnp.inf)
        matrix = matrix.at[r, c].set(0)

        return (matrix, rows, cols), None

    # Run the scan
    (_, row_indices, col_indices), _ = jax.lax.scan(
        scan_fn, (cost_matrix, row_indices, col_indices), jnp.arange(m_size)
    )
    row_indices = row_indices.astype(jnp.int32)
    col_indices = col_indices.astype(jnp.int32)

    return row_indices, col_indices
