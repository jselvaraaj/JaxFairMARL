from datetime import datetime

from config.mappo_config import MAPPOConfig
from train.train_with_single_seed import main

timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
config: MAPPOConfig = MAPPOConfig.create()
assert (
    config.training_config.num_envs > 1
), "Number of environments must be greater than 1 for training"
if config.training_config.seed is None:
    seeds = list(range(config.training_config.num_seeds))
else:
    seeds = list(config.training_config.seed)

assert (
    config.wandb_config.live_logging
), "Live logging must be enabled for sequential training to see wandb logs"

for seed in seeds:
    main(seed, timestamp)
