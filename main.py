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
        "-p", "--project", choices=["mujoco_plane", "mujoco_simple_maze"], required=True,
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
        "-mn", "--model-name", required=False,
        help="Specify the model name (Required for train and evaluate)."
    )

    args = parser.parse_args()

    # Ensure model name is provided for training or evaluation
    if args.action in ["train", "evaluate"] and not args.model_name:
        logger.error("Error: Model name must be specified when using 'train' or 'evaluate'.")
        return

    # Attempt to import the specified project module
    try:
        project_module = importlib.import_module(args.project)
    except ModuleNotFoundError:
        logger.error(f"Error: Project '{args.project}' not found.")
        return

    # Determine function name based on algorithm
    function_name = args.action if args.algorithm is None else f"{args.action}_{args.algorithm}"

    # Retrieve the function dynamically
    try:
        selected_function = getattr(project_module, function_name)
        if args.model_name:
            selected_function(model_name=args.model_name)
        else:
            selected_function()
    except AttributeError:
        logger.error(f"Error: Function '{function_name}' not found in project '{args.project}'.")
    except Exception as e:
        logger.error(f"Error executing function '{function_name}': {e}")

if __name__ == "__main__":
    main()
