from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

import config as cfg
import logging

from mujoco_plane.labyrinth_env import LabyrinthEnv

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def evaluate(model_name: str):
    """
    Evaluate a trained reinforcement learning model by running test episodes and recording evaluation videos.
    The function loads the trained model, executes evaluation episodes, and calculates the success rate.
    """

    EPISODE_LENGTH = 4000  # Maximum steps per evaluation episode
    N_EPISODES = 100  # Number of evaluation episodes
    VIDEO_RECORD_INTERVAL = 10  # Record every nth episode
    VIDEO_FOLDER = cfg.VIDS_DIR  # Directory for storing evaluation videos
    MODEL_PATH = f"{cfg.MODELS_DIR}/{model_name}"  # Path to the trained model

    # Initialize evaluation environment
    eval_env = LabyrinthEnv(episode_length=EPISODE_LENGTH, render_mode='rgb_array')
    eval_env = Monitor(eval_env)  # Monitor for logging stats
    eval_env = RecordVideo(
        eval_env,
        video_folder=VIDEO_FOLDER,
        name_prefix=f"eval_{model_name}",
        episode_trigger=lambda episode: episode % VIDEO_RECORD_INTERVAL == 0  # Record specified episodes
    )

    # Load trained model
    model = SAC.load(MODEL_PATH)

    success_count = 0  # Counter for successful episodes

    # Evaluate model over multiple episodes
    for _ in range(N_EPISODES):
        obs, _ = eval_env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            action = action[0]
            obs, _, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated

            if terminated:
                success_count += 1  # Count successful terminations

    # Cleanup environment
    eval_env.close()

    # Output evaluation results
    logger.info(f"Evaluation completed. Success rate: {success_count}/{N_EPISODES}")