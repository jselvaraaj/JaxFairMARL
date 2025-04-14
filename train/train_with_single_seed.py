import argparse
from datetime import datetime

from algorithm.marl_ppo import experiment_with_single_seed
from config.mappo_config import MAPPOConfig

# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update(
#     "jax_persistent_cache_enable_xla_caches",
#     "xla_gpu_per_fusion_autotune_cache_dir",
# )
# jax.config.update("jax_logging_level", "DEBUG")


def main(seed=0, timestamp=None):
    """Run experiment with the given seed and timestamp."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    config: MAPPOConfig = MAPPOConfig.create()
    assert (
        config.training_config.num_envs > 1
    ), "Number of environments must be greater than 1 for training"
    experiment_with_single_seed(seed, config, timestamp)


if __name__ == "__main__":
    default_timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")

    parser = argparse.ArgumentParser(description="Run experiment with specific seed")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--timestamp", type=str, default=default_timestamp, help="Timestamp"
    )
    args = parser.parse_args()
    main(args.seed, args.timestamp)
