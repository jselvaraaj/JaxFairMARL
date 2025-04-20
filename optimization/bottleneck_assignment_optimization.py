import jax.numpy as jnp
from jaxtyping import Array, Float
from ortools.sat.python import cp_model

# ------------------------------------------------------------
#  Min‑max fair bipartite assignment via Google OR‑Tools CP‑SAT
# ------------------------------------------------------------
#
#  Problem statement
#  -----------------
#  We have m agents and n (≥ m) landmarks with cost matrix C.
#  Each agent must receive exactly one landmark and each landmark may
#  be used at most once.  The objective is *not* to minimise
#  the total cost (Hungarian) but the maximum individual cost
#  across all assignments:
#
#          minimise     z   =  max_{i}  C[i, π(i)]
#
#  This “bottleneck” criterion is the standard definition of
#  min‑max (a.k.a. egalitarian) fairness for one‑to‑one matching.
#
#  Key modelling idea
#  ------------------
#  Introduce a single IntVar z and force      z ≥ C_ij * x_ij
#  for every edge (i,j).  CP‑SAT offers      AddMaxEquality
#  which encodes *all* those inequalities and z == max()   in
#  **one** constraint, eliminating an O(m n) loop.
#
# ------------------------------------------------------------


def minmax_fair_assignment(_cost: Float[Array, "m n"]):
    """
    Solve the min‑max fair assignment problem.

    Parameters
    ----------
    cost : Float[Array, "m n"]
        Rectangular cost matrix cost[i][j] (len = m × n, with n >= m).

    Returns
    -------
    (int, Array[int, "m"], Array[int, "m"])
        Tuple (max_cost, agent_idx, landmark_idx) where
            max_cost  -- the minimised value of the worst individual cost
            For a given pairing i, the agent_idx[i] is the index of the agent
            chosen for landmark_idx[i].
    """
    int_cost = scale_float_cost_matrix_to_int_cost_matrix(_cost)
    m, n = int_cost.shape
    model = cp_model.CpModel()

    # --------------------------------------------------------
    # 1) Decision variables
    # --------------------------------------------------------
    # We create *all* Boolean x_{i,j} in a **single dictionary
    # comprehension** (one loop only):
    #     x[(i,j)] == 1  ↔  worker i is assigned task j.
    x = {(i, j): model.NewBoolVar(f"x_{i}_{j}") for i in range(m) for j in range(n)}

    # Upper bound for z is the largest matrix entry — cheap to compute.
    z = model.NewIntVar(0, jnp.max(int_cost), "z")

    # --------------------------------------------------------
    # 2) Classical assignment constraints
    # --------------------------------------------------------
    # Each worker takes exactly one task (|π(i)| = 1).
    for i in range(m):  # *single* loop
        model.Add(sum(x[i, j] for j in range(n)) == 1)

    # Each task is used by at most one worker (injective π).
    for j in range(n):  # *single* loop
        model.Add(sum(x[i, j] for i in range(m)) <= 1)

    # --------------------------------------------------------
    # 3) Fairness link  ——  one‑liner thanks to AddMaxEquality
    # --------------------------------------------------------
    # The list comprehension builds      cost_ij * x_ij   for every edge.
    # Because x_ij is Boolean, this is still a *linear* term and therefore
    # legal in CP‑SAT.  AddMaxEquality enforces           z == max(list).
    model.AddMaxEquality(
        z, [int_cost[i][j] * x[i, j] for i in range(m) for j in range(n)]
    )

    # --------------------------------------------------------
    # 4) Objective
    # --------------------------------------------------------
    model.Minimize(z)

    # --------------------------------------------------------
    # 5) Solve
    # --------------------------------------------------------
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30  # safeguard: cut after 30 s
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible assignment found")

    # Extract π(i) by scanning each worker’s row for x_{i,j} == 1.
    agent_idx = []
    landmark_idx = []
    for i in range(m):
        for j in range(n):
            if solver.Value(x[i, j]):
                agent_idx.append(i)
                landmark_idx.append(j)
                break

    return solver.Value(z), jnp.asarray(agent_idx), jnp.asarray(landmark_idx)


def scale_float_cost_matrix_to_int_cost_matrix(
    cost: Float[Array, "m n"],
    max_decimals=2,
):
    """
    Converts an m×n float matrix to the smallest‑possible integer matrix
    by finding a common denominator.

    Returns
    -------
    scaled_cost : Float[Array, "m n"]
    """
    factor = 10**max_decimals
    int_cost = jnp.rint(cost * factor).astype(jnp.int32)
    return int_cost


if __name__ == "__main__":
    C = jnp.asarray(
        [
            [90, 76, 75, 70, 50, 74],
            [35, 85, 55, 65, 48, 101],
            [125, 95, 90, 105, 59, 120],
            [45, 110, 95, 115, 104, 83],
            [60, 105, 80, 75, 59, 62],
        ]
    )

    worst, agent_idx, landmark_idx = minmax_fair_assignment(C)
    print("Optimal worst‑case cost:", worst)
    for i, j in enumerate(agent_idx):
        print(f"agent {i} → landmark {j}   (cost = {C[i][j]})")
