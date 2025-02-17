from gymnasium.wrappers import RecordVideo
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from mujoco_plane.labyrinth_env import LabyrinthEnv
from mujoco_plane.tensorboard_integration import TensorboardCallback

def train():
    """ Train the reinforcement learning model """

    # Create vectorized environment for training
    vec_env = make_vec_env(LabyrinthEnv, n_envs=4, env_kwargs={"episode_length": 5000, "render_mode": "rgb_array"})

    # Create a separate logging environment (not vectorized)
    log_env = LabyrinthEnv(episode_length=5000, render_mode='rgb_array')
    log_env = Monitor(log_env)

    model = SAC("MultiInputPolicy", vec_env, verbose=1, tensorboard_log="./output/tensorboard/sac_labyrinth_tensorboard")
    tensor_callback = TensorboardCallback(log_env, render_freq=5000)

    model.learn(total_timesteps=50000, log_interval=100, callback=tensor_callback)
    model.save("./output/models/sac_labyrinth")

    print("Training completed and model saved.")


def evaluate():
    """ Evaluate the trained model and record video """

    # Create separate evaluation environment
    eval_env = LabyrinthEnv(episode_length=1500, render_mode='rgb_array')
    eval_env = Monitor(eval_env)

    # Create separate environment for recording video
    video_env = LabyrinthEnv(episode_length=1500, render_mode="rgb_array")
    video_env = Monitor(video_env)
    video_env = RecordVideo(video_env, video_folder="./output/vids", name_prefix="eval",
                            episode_trigger=lambda x: x % 10 == 0)

    # Load trained model
    model = SAC.load("./output/models/sac_labyrinth")

    # Evaluate model
    n_episodes = 20
    success = 0
    for episode_num in range(n_episodes):
        obs, info = eval_env.reset()
        episode_over = False
        while not episode_over:
            action, _states = model.predict(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            if terminated:
                success += 1
            episode_over = terminated or truncated

    eval_env.close()
    video_env.close()
    print(f"Success rate: {success}/{n_episodes}")
