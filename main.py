import argparse
import subprocess
import importlib
import sys

def view_mujoco(project):
    """Launch Mujoco visualization in an `mjpython` subprocess and wait for it to finish."""
    try:
        process = subprocess.Popen(["mjpython", "-c", f"import {project}; {project}.view()"])
        process.wait()  # Wait for the viewer to close
        sys.exit(0)  # Ensure the main script exits after viewer closes
    except subprocess.CalledProcessError as e:
        print(f"Error running `mjpython`: {e}")
        sys.exit(1)  # Exit with failure code
    except KeyboardInterrupt:
        sys.exit(0)  # Handle Ctrl+C gracefully


def main():
    parser = argparse.ArgumentParser(description="RL Project Launcher")
    parser.add_argument("--project", choices=["mujoco_plane"], help="Specify the RL project version")
    parser.add_argument("--action", choices=["train", "evaluate", "view"], help="Specify the action: train, evaluate or view")

    args = parser.parse_args()

    try:
        project_module = importlib.import_module(args.project)
    except ModuleNotFoundError:
        print(f"Error: Project '{args.project}' not found.")
        return

    if args.action == "view":
        view_mujoco(args.project)  # Call subprocess to start `mjpython` with the correct project // Necessary on MacOs
        return # Exit to avoid checking function_map

    function_map = {
        "mujoco_plane": {
            "train": getattr(project_module, "train"),
            "evaluate": getattr(project_module, "evaluate"),
        }
    }

    selected_function = function_map.get(args.project, {}).get(args.action)
    
    if selected_function:
        try:
            selected_function()
        except Exception as e:
            print(f"Error executing function: {e}")
    else:
        print("Invalid combination of arguments.")
    
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