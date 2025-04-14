import os
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import jax.sharding as sharding
import orbax
import wandb
from flax.training import orbax_utils

from config.config_format_conversion import config_to_dict
from config.mappo_config import MAPPOConfig
from train.train_with_single_seed import main


def callback(bulk_metric, num_steps, config):
    for step in range(num_steps):
        metric = jax.tree.map(lambda leaf: leaf[step], bulk_metric)
        out = metric["actor_network"]
        progress = round(
            (metric["update_steps"] / config.derived_values.num_updates) * 100,
            4,
        )
        update_steps = metric["update_steps"]
        if (
            config.wandb_config.save_model
            and update_steps % config.wandb_config.checkpoint_model_every_update_steps
            == 0
        ):
            dict_config = config_to_dict(config)

            model_artifact = wandb.Artifact(
                "PPO_RNN_Runner_State",
                type="model",
                metadata=dict_config,
            )
            running_script_path = os.path.abspath(".")
            checkpoint_dir = os.path.join(
                running_script_path,
                f"saved_actor/{wandb.run.name}/PPO_Runner_Checkpoint_{progress}",
            )
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(out)
            orbax_checkpointer.save(checkpoint_dir, out, save_args=save_args)
            model_artifact.add_dir(checkpoint_dir)
            wandb.log_artifact(model_artifact)
        wandb.log(
            {
                "returns": metric["returned_episode_returns"][-1, :].mean(),
                "env_step": update_steps
                * config.training_config.num_envs
                * config.training_config.ppo_config.num_steps_per_update,
                **metric["loss"],
            }
        )


if __name__ == "__main__":
    jax.config.update("jax_threefry_partitionable", True)

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
        not config.wandb_config.live_logging
    ), "Live logging must be disabled for parallel training"

    j_seeds = jnp.asarray(seeds)
    if config.SPMD:
        num_devices = jax.device_count()
        print(f"Using {num_devices} seeds since SPMD is enabled")
        j_seeds = j_seeds[:num_devices]
        mesh = jax.make_mesh((num_devices,), ("x",))
        shard = sharding.NamedSharding(mesh, sharding.PartitionSpec("x"))
        j_seeds = jax.device_put(j_seeds, shard)

    out_v = jax.jit(jax.vmap(partial(main, timestamp=timestamp)))(j_seeds)
    jax.block_until_ready(out_v)
    jax.effects_barrier()

    for i, seed in enumerate(seeds):
        # Create a new PyTree containing the i-th slice of each leaf
        out_by_seed = jax.tree.map(lambda leaf: leaf[i], out_v)

        dict_config = config_to_dict(config)

        wandb.init(
            entity=config.wandb_config.entity,
            project=config.wandb_config.project,
            mode=config.wandb_config.mode,
            config=dict_config,
            group=f"experiment_{timestamp}",
            name=f"experiment_{timestamp}_seed_{seed}",
            reinit=True,
        )
        callback(out_by_seed["metric"], config.derived_values.num_updates, config)
        wandb.finish()
