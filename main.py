import argparse

import mujoco_plane


def main():
    parser = argparse.ArgumentParser(description="RL Project Launcher")
    parser.add_argument("version", choices=["mujoco_plane"], help="Specify the RL project version")
    parser.add_argument("mode", choices=["train", "evaluate"], help="Specify the mode: train or evaluate")

    args = parser.parse_args()

    function_map = {
        "mujoco_plane": {
            "train": mujoco_plane.train,
            "evaluate": mujoco_plane.evaluate,
        }
    }

    selected_function = function_map.get(args.version, {}).get(args.mode)

    if selected_function:
        try:
            selected_function()
        except Exception as e:
            print(f"Error executing function: {e}")
    else:
        print("Invalid combination of arguments.")

if __name__ == "__main__":
    main()