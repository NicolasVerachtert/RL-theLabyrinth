from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import config as cfg
import os
import logging

from mujoco_plane.labyrinth_env import LabyrinthEnv
from mujoco_plane.tensorboard_integration import TensorboardCallback

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def train(model_name: str, **kwargs):
    """
    Train a reinforcement learning model using Soft Actor-Critic (SAC) in a custom Labyrinth environment.
    """

    EPISODE_LENGTH = 5_000 # Define the length of each episode (i.e., max number of steps in one episode)
    TOTAL_STEPS = 50_000 # Define how many total steps the agent should train for (across all episodes and environments)
    RENDER_MODE = "rgb_array" # Set the rendering mode (useful for visual debugging, "rgb_array" enables rendering for logs)
    N_ENVS = 4 # Number of parallel environments for training (vectorized environments speed up learning by running multiple instances in parallel)

    # Create a vectorized training environment with multiple instances of LabyrinthEnv
    vec_env = make_vec_env(
        LabyrinthEnv,
        n_envs=N_ENVS,
        env_kwargs={"episode_length": EPISODE_LENGTH, "render_mode": RENDER_MODE}
    )

    # Create a separate single-instance environment for logging purposes
    log_env = LabyrinthEnv(episode_length=EPISODE_LENGTH, render_mode=RENDER_MODE)
    log_env = Monitor(log_env)  # Wrap the environment to log performance statistics

    # Initialize the SAC model with a MultiInputPolicy (handles environments with multiple observation inputs)
    model = SAC(
        policy="MultiInputPolicy",  # Uses a policy that can process multiple input types (e.g., images + numerical data)
        env=vec_env,  # The training environment (vectorized for efficiency)
        verbose=1,  # Verbosity level (1 = print progress updates, 0 = silent, 2 = detailed logs)
        tensorboard_log=f"{cfg.TENSORBOARD_DIR}/{model_name}_tensorboard"
    )

    # Set up a TensorBoard callback to log performance and optionally save rendered videos
    tensor_callback = TensorboardCallback(
        log_env, 
        render_freq=EPISODE_LENGTH, 
        log_video=True
    )

    # Set up a checkpoint callback to save model weights periodically
    os.makedirs(f"{cfg.MODELS_DIR}/{model_name}/checkpoints", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,  # Save the model every 10,000 training steps
        save_path=f"{cfg.MODELS_DIR}/{model_name}/checkpoints"
    )

    # Start the training process
    # log_interval = number of episodes between logging updates
    model.learn(
        total_timesteps=TOTAL_STEPS,  # Total number of steps across all environments
        log_interval=2,  # Log training progress every 2 episodes
        callback=[tensor_callback, checkpoint_callback]
    )

    model.save(f"{cfg.MODELS_DIR}/{model_name}")

    logger.info(f"Training completed and model saved as {cfg.MODELS_DIR}/{model_name}")
