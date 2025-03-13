import importlib
import os
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from mujoco_complex_maze.tensorboard_integration import TensorboardCallback # Assuming this is a custom callback
import config as cfg  # Configuration file
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train_algorithm(model_name: str, env_variant: str, algorithm, total_steps: int, checkpoint_interval: int, episode_length: int, n_envs: int, log_interval: int):
    """
    Train a reinforcement learning model using the specified algorithm in a custom Labyrinth environment.
    :param model_name: Name of the model
    :param env_variant: Environment variant
    :param algorithm: RL algorithm (SAC or PPO)
    :param total_steps: Total training steps
    :param checkpoint_interval: Steps between checkpoints
    :param episode_length: Maximum steps per episode
    :param n_envs: Number of parallel environments
    :param log_interval: Logging interval in steps
    """
    RENDER_MODE = "rgb_array"

    module = importlib.import_module(f"mujoco_complex_maze.labyrinth_env_{env_variant}")
    LabyrinthEnv = getattr(module, "LabyrinthEnv")

    vec_env = make_vec_env(
        LabyrinthEnv,
        n_envs=n_envs,
        env_kwargs={"episode_length": episode_length, "render_mode": RENDER_MODE}
    )

    log_env = LabyrinthEnv(episode_length=episode_length, render_mode=RENDER_MODE)
    log_env = Monitor(log_env)  # Wrap the environment for logging

    model = algorithm(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        tensorboard_log=f"{cfg.TENSORBOARD_DIR}/{model_name}_tensorboard",
        device="cuda"
    )

    tensor_callback = TensorboardCallback(
        log_env,
        render_freq=episode_length,
        log_interval=20,
        log_video=True
    )

    os.makedirs(f"{cfg.MODELS_DIR}/{model_name}/checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_interval,
        save_path=f"{cfg.MODELS_DIR}/{model_name}/checkpoints"
    )

    model.learn(
        total_timesteps=total_steps,
        log_interval=log_interval,
        callback=[tensor_callback, checkpoint_callback]
    )

    model.save(f"{cfg.MODELS_DIR}/{model_name}")
    logger.info(f"Training completed and model saved as {cfg.MODELS_DIR}/{model_name}")

def train(model_name: str, env_variant: str, algorithm: str):
    """
    Train a reinforcement learning model using either SAC or PPO.
    :param model_name: Name of the model
    :param env_variant: Environment variant
    :param algorithm: Algorithm name ('SAC' or 'PPO')
    """
    config = {
        "SAC": {
            "algorithm": SAC,
            "total_steps": 10_000_000,
            "checkpoint_interval": 1_000_000,
            "episode_length": 4_000,
            "n_envs": 6,
            "log_interval": 10_000
        },
        "PPO": {
            "algorithm": PPO,
            "total_steps": 10_000_000,
            "checkpoint_interval": 1_000_000,
            "episode_length": 4_000,
            "n_envs": 6,
            "log_interval": 10_000
        }
    }

    if algorithm not in config:
        raise ValueError("Algorithm must be either 'SAC' or 'PPO'")

    params = config[algorithm]
    train_algorithm(model_name, env_variant, **params)
