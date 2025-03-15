import logging
import cv2
import importlib
from stable_baselines3 import SAC, PPO
import config as cfg


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def demo(model_name: str, env_variant: str, algorithm: str):
    possible_algos = {
        "SAC": SAC,
        "PPO": PPO,
    }

    if algorithm not in possible_algos:
        raise ValueError("Algorithm must be either 'SAC' or 'PPO'")

    algo = possible_algos[algorithm]

    
    RENDER_MODE = "rgb_array"
    N_EPISODES = 5

    module = importlib.import_module(f"mujoco_complex_maze.labyrinth_env_{env_variant}")
    LabyrinthEnv = getattr(module, "LabyrinthEnv")

    demo_env = LabyrinthEnv(episode_length=4000, render_mode=RENDER_MODE, evaluation_vid=True, demo=True)
    model = algo.load(f"{cfg.MODELS_DIR}/{model_name}")

    for _ in range(N_EPISODES):
        obs, _ = demo_env.reset()
        episode_over = False
        while not episode_over:
            action, _ = model.predict(obs)
            
            obs, _, terminated, truncated, _ = demo_env.step(action)
            should_exit = demo_env.render()
            if should_exit.dtype == bool and should_exit:
                break

            episode_over = terminated or truncated

    demo_env.close()
    cv2.destroyAllWindows()

