import numpy as np  # Use numpy for pairwise distance calculation helper
from jax import random

from envs.utils import sample_points


def test_sample_points_min_distance():
    """Tests the minimum distance constraint."""
    key = random.PRNGKey(1)
    num_points = 10
    min_dist = 0.2
    bounds = (0, 1)
    points = sample_points(num_points, key, min_dist, bounds)

    assert points.shape == (num_points, 2)

    # Calculate pairwise distances
    points_np = np.array(points)  # Convert to numpy for easier distance calculation
    diffs = points_np[:, np.newaxis, :] - points_np[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))

    # Set diagonal (distance to self) to infinity to ignore it in min check
    np.fill_diagonal(dists, np.inf)

    assert np.all(dists >= min_dist)
