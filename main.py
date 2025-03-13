import argparse
import importlib
from config import setup_environment

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    """
    Command-line interface for launching different RL project actions such as training, evaluation, and viewing.
    Includes an optional argument for specifying the RL algorithm.
    Uses an .env file for configurable settings such as output directories.
    """

    # Load .env and setup environment
    setup_environment()

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="RL Project Launcher")
    parser.add_argument(
        "-p", "--project", choices=["mujoco_plane", "mujoco_simple_maze", "mujoco_complex_maze"], required=True,
        help="Specify the RL project version"
    )
    parser.add_argument(
        "-a", "--action", choices=["train", "evaluate", "view"], required=True,
        help="Specify the action: train, evaluate, or view"
    )
    parser.add_argument(
        "-alg", "--algorithm", default=None,
        help="Specify the RL algorithm (e.g., SAC, PPO). If not specified, the module's default will be used."
    )
    parser.add_argument(
        "-env", "--environment", default=None,
        help="Specify the environment variant. If not specified, the module's default will be used."
    )
    parser.add_argument(
        "-mn", "--model-name", required=False,
        help="Specify the model name (Required for train and evaluate)."
    )

    args = parser.parse_args()

    # Ensure model name is provided for training or evaluation
    if args.action in ["train", "evaluate"] and not args.model_name:
        logger.error("Error: Model name must be specified when using 'train' or 'evaluate'.")
        return

    # Ensure algorithm and environment variant are provided for training in the complex maze
    if args.action == "train" and args.project == "mujoco_complex_maze" and (not args.environment or not args.algorithm):
        logger.error("Error: When training in Mujoco Complex Maze, you must specify the RL algorithm (e.g., SAC, PPO) and the environment variant.")
        return

    # Attempt to import the specified project module
    try:
        project_module = importlib.import_module(args.project)
    except ModuleNotFoundError as e:
        logger.error(f"Error: Project '{args.project}' not found.: {e}")
        return

    # Retrieve the function dynamically
    try:
        selected_function = getattr(project_module, args.action)
        if args.model_name:
            selected_function(
                model_name = args.model_name,
                env_variant = args.environment, 
                algorithm = args.algorithm)
        else:
            selected_function()
    except AttributeError:
        logger.error(f"Error: Function '{args.action}' not found in project '{args.project}'.")
    except Exception as e:
        logger.error(f"Error executing function '{args.action}': {e}")

if __name__ == "__main__":
    main()
